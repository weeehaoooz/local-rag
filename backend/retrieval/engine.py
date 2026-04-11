import sniffio_compat
sniffio_compat.apply()

import nest_asyncio
# Apply nest_asyncio to handle nested event loops in llama-index
nest_asyncio.apply()

import os
import asyncio
import logging
import time
import numpy as np
from dotenv import load_dotenv

from llama_index.core import (
    PropertyGraphIndex, StorageContext, Settings, load_index_from_storage,
)
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Load environment variables
load_dotenv()

from config import DEFAULT_LLM, DEFAULT_EMBED
from retrieval.services.router import RouterService, QueryType
from retrieval.services.graph_service import GraphService
from retrieval.services.fusion import RankFusionService
from retrieval.services.formatter import ContextFormatter
from retrieval.services.evaluator import ReflectionService
from retrieval.services.orchestrator import ToolOrchestrator
from retrieval.services.transformer import QueryTransformer
from retrieval.services.decomposer import QueryDecomposer
from research.web_searcher import WebSearcher
from research.searcher import ResearchSearcher

logger = logging.getLogger(__name__)


class HybridEngine:
    """
    KG-RAG query engine that orchestrates retrieval across:
      1. PropertyGraphIndex (Neo4j KG)
      2. VectorStoreIndex (Semantic Search)
      3. SummaryIndex (Document Overviews)
      4. CommunitySummaries (GraphRAG Global Search)

    It delegates specialized logic to helper services.
    """

    def __init__(self):
        # ── LLM & Embedding ───────────────────────────────────────────
        self.llm = Ollama(
            model=DEFAULT_LLM,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            request_timeout=720.0,
            context_window=int(os.getenv("OLLAMA_CONTEXT_WINDOW", "8192")),
            additional_kwargs={
                "num_ctx": int(os.getenv("OLLAMA_CONTEXT_WINDOW", "8192")),
            },
        )
        self.embed_model = OllamaEmbedding(
            model_name=DEFAULT_EMBED,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            request_timeout=360.0,
        )

        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        # ── Storage & Indices ─────────────────────────────────────────
        self.graph_store = Neo4jPropertyGraphStore(
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),
            url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            refresh_schema=False,
        )

        self.kg_index = PropertyGraphIndex.from_existing(
            property_graph_store=self.graph_store,
            llm=self.llm,
            embed_model=self.embed_model,
        )

        # ── Initialize Services ───────────────────────────────────────
        self.router = RouterService(self.llm)
        self.graph_service = GraphService(self.llm, self.graph_store)
        self.fusion_service = RankFusionService()
        self.formatter = ContextFormatter()
        self.reflection = ReflectionService(self.llm, fail_open=True)
        self.orchestrator = ToolOrchestrator(self.llm, fail_open=True)
        self.transformer = QueryTransformer(self.llm)
        self.decomposer = QueryDecomposer(self.llm)
        self.web_searcher = WebSearcher(max_results=3)
        self.arxiv_searcher = ResearchSearcher(max_results_per_query=3)

        # ── Load Persisted Indices ────────────────────────────────────
        self._load_indices()
        print("HybridEngine ready.")

    def _load_indices(self):
        from config import VECTOR_DIR, BM25_DIR, SUMMARY_DIR
        print(f"Loading indexes from {VECTOR_DIR}/..")
        try:
            vector_ctx = StorageContext.from_defaults(persist_dir=VECTOR_DIR)
            self.vector_index = load_index_from_storage(vector_ctx)
            print("  ✓ Vector index loaded")
            
            # --- Dimension Check ---
            try:
                # Check current model dimension
                current_dim = len(self.embed_model.get_query_embedding("test"))
                
                # Try to find a stored embedding to check its dimension
                # We can peek into the vector store
                vector_store = self.vector_index.vector_store
                stored_dim = None
                
                # Different vector stores have different ways of exposing this
                # For SimpleVectorStore (standard for local), it's in data.embedding_dict
                if hasattr(vector_store, "_data") and hasattr(vector_store._data, "embedding_dict"):
                    emb_dict = vector_store._data.embedding_dict
                    if emb_dict:
                        stored_dim = len(next(iter(emb_dict.values())))

                if stored_dim and stored_dim != current_dim:
                    print("\n" + "!"*60)
                    print("CRITICAL: EMBEDDING DIMENSION MISMATCH DETECTED")
                    print(f"Stored Index: {stored_dim} dimensions")
                    print(f"Current Model: {current_dim} dimensions ({self.embed_model.model_name})")
                    print("!"*60)
                    print("\nTo fix this error, you must either:")
                    print(f"1. Switch your DEFAULT_EMBED in config.py to a model with {stored_dim} dims.")
                    print("2. Delete your 'storage/' directory and re-run your indexing.")
                    print("!"*60 + "\n")
            except Exception as dim_err:
                # Fail open for dimension check to avoid blocking startup
                print(f"  ? Could not verify index dimensions: {dim_err}")
                
        except Exception as e:
            print(f"  ✗ Vector index not available: {e}")
            self.vector_index = None

        try:
            summary_ctx = StorageContext.from_defaults(persist_dir=SUMMARY_DIR)
            self.summary_index = load_index_from_storage(summary_ctx)
            print("  ✓ Summary index loaded")
        except Exception as e:
            print(f"  ✗ Summary index not available: {e}")
            self.summary_index = None

        # BM25
        self.bm25_retriever = None
        try:
            from llama_index.retrievers.bm25 import BM25Retriever
            bm25_dir = BM25_DIR
            if os.path.exists(bm25_dir):
                self.bm25_retriever = BM25Retriever.from_persist_dir(bm25_dir)
                print("  ✓ BM25 index loaded")
        except Exception as e:
            print(f"  ✗ BM25 index not available: {e}")

        # Community Summarizer
        try:
            from indexing.community import CommunitySummarizer
            self.community_summarizer = CommunitySummarizer(self.graph_store, llm=self.llm)
        except Exception:
            self.community_summarizer = None

    def get_suggestions(self, limit: int = 4) -> list[str]:
        """Suggest quick prompts based on the Knowledge Graph."""
        suggestions = []
        try:
            entity_query = """
            MATCH (n:Entity)
            WITH n, size((n)--()) AS degree
            ORDER BY degree DESC
            LIMIT 10
            RETURN n.name AS name
            """
            results = self.graph_store.structured_query(entity_query)
            if isinstance(results, list):
                for r in results:
                    name = r.get("name") if isinstance(r, dict) else (r[0] if r else None)
                    if name:
                        suggestions.append(f"Tell me about {name}")

            if not suggestions:
                return ["What is in the knowledge base?", "Summarize the key themes."]

            return suggestions[:limit]
        except Exception:
            return ["Tell me about the latest documents."]

    async def get_context_async(self, query: str, plan: 'ToolPlan' = None) -> dict:
        """
        Retrieves context using parallel execution of selected tools.
        """
        if not plan:
            plan = await self.orchestrator.analyze_request(query)
            
        tools = plan.tools
        query_type = plan.fallback_query_type
        entity_hints = plan.keywords

        if plan.is_generic and not tools:
            return {
                "graph_context": [], "vector_context": [], "summary_context": [],
                "community_context": [], "sources": [], "query_type": query_type,
                "tools_used": [], "orchestrator_rationale": plan.rationale,
            }

        all_sources = []
        def _to_dicts(nodes):
            res = []
            for n in nodes:
                actual_node = getattr(n, "node", n)
                meta = getattr(actual_node, "metadata", {})
                title = meta.get("title", meta.get("file_name", "Unknown"))
                res.append({"text": actual_node.get_content(), "source": title, "metadata": meta})
            return res

        # Define parallel tasks
        tasks = []
        task_names = []

        # 1. Vector Search Task (includes HyDE)
        async def vector_task():
            if not self.vector_index: return []
            v_query = await self.transformer.generate_hyde_document(query)
            nodes = await self.vector_index.as_retriever(similarity_top_k=5).aretrieve(v_query)
            processed = _to_dicts(nodes)
            all_sources.extend(self.formatter.extract_sources(nodes))
            return processed

        if "vector_search" in tools:
            tasks.append(vector_task())
            task_names.append("vector")

        # 2. BM25 Search Task
        async def bm25_task():
            if not self.bm25_retriever: return []
            loop = asyncio.get_event_loop()
            nodes = await loop.run_in_executor(None, self.bm25_retriever.retrieve, query)
            processed = _to_dicts(nodes)
            all_sources.extend(self.formatter.extract_sources(nodes))
            return processed

        if "vector_search" in tools and self.bm25_retriever:
            tasks.append(bm25_task())
            task_names.append("bm25")

        # 3. Graph Search Task
        async def graph_task():
            kg_retriever = self.kg_index.as_retriever(include_text=True, similarity_top_k=5)
            nodes = await kg_retriever.aretrieve(query)
            processed = _to_dicts(nodes)
            all_sources.extend(self.formatter.extract_sources(nodes))
            return processed

        if "graph_search" in tools:
            tasks.append(graph_task())
            task_names.append("graph")

        # 4. Summary Search Task
        async def summary_task():
            nodes = await self.summary_index.as_retriever().aretrieve(query)
            processed = [(n.get_content(), "Summary") for n in nodes[:2]]
            all_sources.extend(self.formatter.extract_sources(nodes))
            return processed

        if "summary_search" in tools and self.summary_index:
            tasks.append(summary_task())
            task_names.append("summary")

        # 5. Community Search Task
        async def community_task():
            communities = self.community_summarizer.get_relevant_summaries(query, top_k=3)
            return [(c.get("summary", ""), f"Community {c.get('community_id')}") for c in communities]

        if "community_search" in tools and self.community_summarizer:
            tasks.append(community_task())
            task_names.append("community")

        # 6. Web/ArXiv Search Tasks
        async def web_task():
            results = self.web_searcher.search_text([query])
            return [(r["snippet"], r["title"], r["link"]) for r in results]

        if "web_search" in tools and self.web_searcher:
            tasks.append(web_task())
            task_names.append("web")

        async def arxiv_task():
            results = self.arxiv_searcher.search([query])
            return [(r["summary"], r["title"], r["id"]) for r in results]

        if "arxiv_search" in tools and self.arxiv_searcher:
            tasks.append(arxiv_task())
            task_names.append("arxiv")

        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Unpack results
        res_map = {name: val for name, val in zip(task_names, results) if not isinstance(val, Exception)}
        
        vector_candidates = res_map.get("vector", [])
        bm25_candidates = res_map.get("bm25", [])
        graph_candidates = res_map.get("graph", [])
        summary_results = res_map.get("summary", [])
        community_results = res_map.get("community", [])
        
        # Handle Web/ArXiv results separately (they append to all_sources)
        fused_context = []
        for r in res_map.get("web", []):
            fused_context.append((r[0], r[1]))
            all_sources.append({"title": r[1], "category": "Web", "file": r[2]})
        for r in res_map.get("arxiv", []):
            fused_context.append((r[0], r[1]))
            all_sources.append({"title": r[1], "category": "ArXiv", "file": r[2]})

        # 7. Fusion & Graph Post-processing
        expanded_entities = []
        if vector_candidates or bm25_candidates or graph_candidates:
            fusion_input = [l for l in [vector_candidates, bm25_candidates, graph_candidates] if l]
            if fusion_input:
                fused_results = self.fusion_service.reciprocal_rank_fusion(fusion_input)
                top_fused = fused_results[:5]
                fused_context.extend([(c["text"], c["source"]) for c in top_fused])
                
                # Expand seeds from keywords + graph hits
                seeds = entity_hints + [n['metadata'].get('name') for n in graph_candidates if n['metadata'].get('name')]
                if seeds:
                    expanded_entities = await self.graph_service.expand_graph_context(list(set(seeds)), max_hops=1)
                
                # Hybrid Traversal
                traversal = self.graph_service.hybrid_graph_traversal(top_fused)
                expanded_entities.extend(traversal.get("entities", []))

        # 8. Entity Summarization
        summarized_graph = []
        if graph_candidates or expanded_entities:
            summarized_graph = await self.graph_service.summarize_entity_context(
                [(c["text"], c["source"]) for c in graph_candidates],
                expanded_entities
            )

        return {
            "graph_context": summarized_graph,
            "vector_context": fused_context,
            "summary_context": summary_results,
            "community_context": community_results,
            "sources": all_sources,
            "query_type": query_type,
            "tools_used": tools,
            "orchestrator_rationale": plan.rationale,
        }

        return {
            "graph_context": summarized_graph,
            "vector_context": fused_context,
            "summary_context": summary_context,
            "community_context": community_context,
            "sources": all_sources,
            "query_type": query_type,
            "tools_used": tools,
            "orchestrator_rationale": plan.rationale,
        }

    def _collect_context_texts(self, context: dict) -> list[str]:
        """Flatten the composite context dict into a plain list of text strings."""
        chunks: list[str] = []
        for text, _ in context.get("graph_context", []):
            if text:
                chunks.append(text)
        for text, _ in context.get("vector_context", []):
            if text:
                chunks.append(text)
        for text, _ in context.get("summary_context", []):
            if text:
                chunks.append(text)
        for text, _ in context.get("community_context", []):
            if text:
                chunks.append(text)
        return chunks

    def _build_prompt(self, context: dict, user_message: str, system_prompt: str) -> str:
        """Compose the full LLM prompt from context sections."""
        prompt_parts = []
        if context["community_context"]:
            prompt_parts.append("COMMUNITY SUMMARIES:\n" + "\n".join(f"- {t}" for t, s in context["community_context"]))
        if context["graph_context"]:
            prompt_parts.append("KNOWLEDGE GRAPH:\n" + "\n".join(f"- {t}" for t, s in context["graph_context"]))
        if context["vector_context"]:
            prompt_parts.append("DOCUMENT EXCERPTS:\n" + "\n".join(f"- {t} (Source: {s})" for t, s in context["vector_context"]))

        context_str = "\n\n".join(prompt_parts) if prompt_parts else "No relevant context found."
        return f"{system_prompt}\n\nContext:\n{context_str}\n\nUser: {user_message}\nAssistant:"

    async def chat_async(
        self,
        user_message: str,
        mode: str = "fast",
        history: list = None,
        system_prompt: str = "",
        max_reflection_loops: int = 2,
    ) -> dict:
        """
        Async main entry point for chat interaction.
        """
        reflection_loops = 0
        retrieval_grade_result = "pass"
        answer_grade_result = "grounded"

        # 1. Consolidated Analysis: Coref + Decomposition + Tools + Keywords
        history = history or []
        plan = await self.orchestrator.analyze_request(user_message, history)
        resolved_query = plan.resolved_query
        sub_queries = plan.sub_queries
        
        print(f"  [Agent] Resolved: {resolved_query} | Sub-queries: {sub_queries}")

        # Container for pooled contexts
        all_context = {
            "graph_context": [], "vector_context": [], "summary_context": [],
            "community_context": [], "sources": [], "tools_used": [],
            "orchestrator_rationale": plan.rationale,
            "query_type": plan.fallback_query_type
        }
        all_context_chunks = []

        # ── Parallel Retrieval & Processing ─────────────────────────────
        async def process_sub_query(sq):
            nonlocal reflection_loops, retrieval_grade_result
            sq_context = await self.get_context_async(sq, plan=plan)
            sq_chunks = self._collect_context_texts(sq_context)

            # Grading loop per sub-query (SKIPPED in fast mode)
            if mode != "fast":
                for loop in range(max_reflection_loops):
                    r_grade = await self.reflection.grade_retrieval(sq, sq_chunks)
                    if r_grade.relevant:
                        break
                    retrieval_grade_result = "fail"
                    reflection_loops += 1
                    if loop < max_reflection_loops - 1:
                        sq = await self.reflection.rewrite_query(sq, r_grade.reason)
                        sq_context = await self.get_context_async(sq, plan=plan)
                        sq_chunks = self._collect_context_texts(sq_context)
            
            return sq_context, sq_chunks

        # Run all sub-queries in parallel
        sub_results = await asyncio.gather(*[process_sub_query(sq) for sq in sub_queries])

        for sq_context, sq_chunks in sub_results:
            # Merge results into pooled context
            all_context["graph_context"].extend(sq_context.get("graph_context", []))
            all_context["vector_context"].extend(sq_context.get("vector_context", []))
            all_context["summary_context"].extend(sq_context.get("summary_context", []))
            all_context["community_context"].extend(sq_context.get("community_context", []))

            for src in sq_context.get("sources", []):
                if src not in all_context["sources"]:
                    all_context["sources"].append(src)

            for t in sq_context.get("tools_used", []):
                if t not in all_context["tools_used"]:
                    all_context["tools_used"].append(t)

            all_context_chunks.extend(sq_chunks)

        # ── Answer generation ────────────────────────────────────────────
        full_prompt = self._build_prompt(all_context, user_message, system_prompt)
        
        # Track time for TPS
        start_t = time.time()
        response = await self.llm.acomplete(full_prompt)
        end_t = time.time()
        answer_text = response.text
        
        ans_duration = end_t - start_t
        completion_tkns = len(answer_text) // 4
        
        # If response has a raw dict with eval stats from ollama, use it
        if hasattr(response, "raw") and isinstance(response.raw, dict) and "eval_count" in response.raw:
            completion_tkns = response.raw.get("eval_count", completion_tkns)
        
        tps = completion_tkns / ans_duration if ans_duration > 0 else 0.0

        # ── Answer grounding check (SKIPPED in fast mode) ────────────────
        if mode != "fast":
            a_grade = await self.reflection.grade_answer(user_message, all_context_chunks, answer_text)
            print(f"  [Reflection] Answer grade: grounded={a_grade.grounded} — {a_grade.reason}")

            if not a_grade.grounded and reflection_loops < max_reflection_loops:
                answer_grade_result = "ungrounded"
                reflection_loops += 1
                corrective_query = await self.reflection.rewrite_query(
                    user_message,
                    failure_reason=f"The answer was ungrounded. Reason: {a_grade.reason}."
                )
                corrective_context = await self.get_context_async(corrective_query, plan=plan)
                corrective_chunks = self._collect_context_texts(corrective_context)
                corrective_prompt = self._build_prompt(corrective_context, user_message, system_prompt)
                corrective_response = await self.llm.acomplete(corrective_prompt)
                final_a_grade = await self.reflection.grade_answer(user_message, corrective_chunks, corrective_response.text)
                if final_a_grade.grounded:
                    answer_text = corrective_response.text
                    all_context = corrective_context
                    answer_grade_result = "grounded"
            else:
                answer_grade_result = "grounded" if a_grade.grounded else "ungrounded"
        else:
            answer_grade_result = "skipped"

        return {
            "response": answer_text,
            "sources": all_context["sources"],
            "query_type": all_context["query_type"].value if all_context.get("query_type") else "HYBRID",
            "suggested_prompts": self.get_suggestions(),
            # ── Reflection diagnostics ─────────────────────────────────
            "reflection_loops": reflection_loops,
            "retrieval_grade": retrieval_grade_result,
            "answer_grade": answer_grade_result,
            # ── Orchestrator diagnostics ───────────────────────────────
            "tools_used": all_context.get("tools_used", []),
            "orchestrator_rationale": all_context.get("orchestrator_rationale", ""),
            "sub_queries": sub_queries,
            "stats": {
                "tps": round(tps, 1),
                "context_utilization": min(1.0, (len(full_prompt) // 4 + completion_tkns) / 8192.0),
                "prompt_tokens": len(full_prompt) // 4,
                "completion_tokens": completion_tkns,
                "total_tokens": len(full_prompt) // 4 + completion_tkns,
                "context_window": 8192
            }
        }

    async def chat_stream_status_async(
        self,
        user_message: str,
        mode: str = "fast",
        history: list = None,
        system_prompt: str = "",
        max_reflection_loops: int = 2,
    ):
        import json

        def _yield_status(msg: str, tokens: int = 0):
            return "data: " + json.dumps({"type": "status", "message": msg, "tokens": tokens}) + "\n\n"

        def _yield_final(data: dict):
            return "data: " + json.dumps({"type": "done", "response": data}) + "\n\n"

        yield _yield_status("Analyzing request...")
        history = history or []
        plan = await self.orchestrator.analyze_request(user_message, history)
        resolved_query = plan.resolved_query
        sub_queries = plan.sub_queries

        all_context = {
            "graph_context": [], "vector_context": [], "summary_context": [],
            "community_context": [], "sources": [], "tools_used": [],
            "orchestrator_rationale": plan.rationale,
            "query_type": plan.fallback_query_type
        }
        all_context_chunks = []
        reflection_loops = 0
        retrieval_grade_result = "pass" if mode == "fast" else "ungraded"
        answer_grade_result = "grounded" if mode == "fast" else "ungraded"

        async def process_sq_status(sq):
            nonlocal reflection_loops, retrieval_grade_result
            # Status yielding is harder from inside a gather, so we'll just yield start
            sq_context = await self.get_context_async(sq, plan=plan)
            sq_chunks = self._collect_context_texts(sq_context)

            if mode != "fast":
                for loop_idx in range(max_reflection_loops):
                    r_grade = await self.reflection.grade_retrieval(sq, sq_chunks)
                    if r_grade.relevant:
                        break
                    retrieval_grade_result = "fail"
                    reflection_loops += 1
                    if loop_idx < max_reflection_loops - 1:
                        sq = await self.reflection.rewrite_query(sq, r_grade.reason)
                        sq_context = await self.get_context_async(sq, plan=plan)
                        sq_chunks = self._collect_context_texts(sq_context)
            return sq_context, sq_chunks

        yield _yield_status(f"Retrieving context for {len(sub_queries)} queries...")
        sub_results = await asyncio.gather(*[process_sq_status(sq) for sq in sub_queries])

        for sq_context, sq_chunks in sub_results:
            all_context["graph_context"].extend(sq_context.get("graph_context", []))
            all_context["vector_context"].extend(sq_context.get("vector_context", []))
            all_context["summary_context"].extend(sq_context.get("summary_context", []))
            all_context["community_context"].extend(sq_context.get("community_context", []))
            for src in sq_context.get("sources", []):
                if src not in all_context["sources"]:
                    all_context["sources"].append(src)
            for t in sq_context.get("tools_used", []):
                if t not in all_context["tools_used"]:
                    all_context["tools_used"].append(t)
            all_context_chunks.extend(sq_chunks)

        full_prompt = self._build_prompt(all_context, user_message, system_prompt)

        # Generation phase
        yield _yield_status("Generating answer...", tokens=0)
        start_t = time.time()

        response_gen = await self.llm.astream_complete(full_prompt)
        answer_text = ""
        token_count = 0
        async for chunk in response_gen:
            answer_text += chunk.delta
            token_count += 1
            if token_count % 5 == 0:
                yield _yield_status("Generating answer...", tokens=token_count)

        end_t = time.time()
        ans_duration = end_t - start_t
        tps = token_count / ans_duration if ans_duration > 0 else 0.0

        if mode != "fast":
            yield _yield_status("Evaluating groundedness...")
            a_grade = await self.reflection.grade_answer(user_message, all_context_chunks, answer_text)

            if not a_grade.grounded and reflection_loops < max_reflection_loops:
                answer_grade_result = "ungrounded"
                reflection_loops += 1
                yield _yield_status("Answer ungrounded, correcting...")

                corrective_query = await self.reflection.rewrite_query(
                    user_message,
                    failure_reason=f"The answer was ungrounded. Reason: {a_grade.reason}."
                )
                corrective_context = await self.get_context_async(corrective_query, plan=plan)
                corrective_chunks = self._collect_context_texts(corrective_context)
                corrective_prompt = self._build_prompt(corrective_context, user_message, system_prompt)

                yield _yield_status("Generating answer...", tokens=0)
                cor_gen = await self.llm.astream_complete(corrective_prompt)
                cor_text = ""
                cor_tokens = 0
                async for chunk in cor_gen:
                    cor_text += chunk.delta
                    cor_tokens += 1
                    if cor_tokens % 5 == 0:
                        yield _yield_status("Generating answer...", tokens=cor_tokens)

                final_a_grade = await self.reflection.grade_answer(user_message, corrective_chunks, cor_text)
                if final_a_grade.grounded:
                    answer_text = cor_text
                    all_context = corrective_context
                    answer_grade_result = "grounded"
            else:
                answer_grade_result = "grounded" if a_grade.grounded else "ungrounded"
        else:
            answer_grade_result = "skipped"

        final_response = {
            "response": answer_text,
            "sources": all_context["sources"],
            "query_type": all_context["query_type"].value if all_context.get("query_type") else "HYBRID",
            "suggested_prompts": self.get_suggestions(),
            "reflection_loops": reflection_loops,
            "retrieval_grade": retrieval_grade_result,
            "answer_grade": answer_grade_result,
            "tools_used": all_context.get("tools_used", []),
            "orchestrator_rationale": all_context.get("orchestrator_rationale", ""),
            "sub_queries": sub_queries,
            "stats": {
                "tps": round(tps, 1),
                "context_utilization": min(1.0, (len(full_prompt) // 4 + token_count) / 8192.0),
                "prompt_tokens": len(full_prompt) // 4,
                "completion_tokens": token_count,
                "total_tokens": len(full_prompt) // 4 + token_count,
                "context_window": 8192
            }
        }

        yield _yield_final(final_response)

    def close(self):
        try:
            self.graph_store.close()
        except Exception:
            pass