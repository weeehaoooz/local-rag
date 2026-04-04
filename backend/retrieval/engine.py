import os
import json
import logging
from dotenv import load_dotenv
import nest_asyncio

# Apply nest_asyncio to handle nested event loops in llama-index
nest_asyncio.apply()

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

    def get_context(self, query: str) -> dict:
        """Retrieve and fuse context from all indices based on Orchestrator plan."""
        # NEW ORCHESTRATOR LOGIC
        plan = self.orchestrator.plan_tools(query)
        tools = plan.tools
        query_type = plan.fallback_query_type
        print(f"  [Orchestrator] Tools: {tools} | Fallback: {query_type.value} | Rationale: {plan.rationale}")

        # Fallback keyword extraction for graph seed expansion
        _, entity_hints = self.router.classify_query(query)

        all_sources = []
        
        def _to_dicts(nodes):
            res = []
            for n in nodes:
                actual_node = getattr(n, "node", n)
                meta = getattr(actual_node, "metadata", {})
                title = meta.get("title", meta.get("file_name", "Unknown"))
                res.append({"text": actual_node.get_content(), "source": title, "metadata": meta})
            return res

        # 1. Semantic (Vector) & Lexical (BM25)
        vector_candidates = []
        if self.vector_index and "vector_search" in tools:
            # Apply HyDE specifically for local semantic search
            vector_query = self.transformer.generate_hyde_document(query)
            nodes = self.vector_index.as_retriever(similarity_top_k=5).retrieve(vector_query)
            vector_candidates = _to_dicts(nodes)
            all_sources.extend(self.formatter.extract_sources(nodes))

        bm25_candidates = []
        if self.bm25_retriever and "vector_search" in tools:
            nodes = self.bm25_retriever.retrieve(query)
            bm25_candidates = _to_dicts(nodes)
            all_sources.extend(self.formatter.extract_sources(nodes))

        # 2. Knowledge Graph
        graph_candidates = []
        expanded_entities = []
        if "graph_search" in tools:
            kg_retriever = self.kg_index.as_retriever(include_text=True, similarity_top_k=5)
            nodes = kg_retriever.retrieve(query)
            graph_candidates = _to_dicts(nodes)
            all_sources.extend(self.formatter.extract_sources(nodes))
            
            # Seed expansion
            seeds = entity_hints + [n['metadata'].get('name') for n in graph_candidates if n['metadata'].get('name')]
            if seeds:
                expanded_entities = self.graph_service.expand_graph_context(list(set(seeds)), max_hops=1)

        # 3. Fusion & Hybrid Traversal
        fused_context = []
        if "vector_search" in tools or "graph_search" in tools:
            fusion_input = [l for l in [vector_candidates, bm25_candidates, graph_candidates] if l]
            if fusion_input:
                fused_results = self.fusion_service.reciprocal_rank_fusion(fusion_input)
                top_fused = fused_results[:5]
                fused_context = [(c["text"], c["source"]) for c in top_fused]
                
                # Expand from hits
                traversal = self.graph_service.hybrid_graph_traversal(top_fused)
                expanded_entities.extend(traversal.get("entities", []))

        # 4. Global Context (Summaries & Communities)
        summary_context = []
        if self.summary_index and "summary_search" in tools:
            nodes = self.summary_index.as_retriever().retrieve(query)
            summary_context = [(n.get_content(), "Summary") for n in nodes[:2]]
            all_sources.extend(self.formatter.extract_sources(nodes))

        community_context = []
        if self.community_summarizer and "community_search" in tools:
            communities = self.community_summarizer.get_relevant_summaries(query, top_k=3)
            for c in communities:
                community_context.append((c.get("summary", ""), f"Community {c.get('community_id')}"))

        # 5. Entity Summarization
        summarized_graph = self.graph_service.summarize_entity_context(
            [(c["text"], c["source"]) for c in graph_candidates],
            expanded_entities
        )
        
        # 6. Web & ArXiv Context (NEW)
        if "web_search" in tools and self.web_searcher:
            try:
                print("  [Orchestrator] Running web_search...")
                web_results = self.web_searcher.search_text([query])
                for r in web_results:
                    fused_context.append((r["snippet"], r["title"]))
                    all_sources.append({"title": r["title"], "category": "Web", "file": r["link"]})
            except Exception as e:
                print(f"  [Orchestrator] web_search failed: {e}")
                
        if "arxiv_search" in tools and self.arxiv_searcher:
            try:
                print("  [Orchestrator] Running arxiv_search...")
                arxiv_results = self.arxiv_searcher.search([query])
                for r in arxiv_results:
                    fused_context.append((r["summary"], r["title"]))
                    all_sources.append({"title": r["title"], "category": "ArXiv", "file": r["id"]})
            except Exception as e:
                print(f"  [Orchestrator] arxiv_search failed: {e}")

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
        """
        Flatten the composite context dict into a plain list of text strings
        for use by the ReflectionService graders.
        """
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

    def chat(
        self,
        user_message: str,
        mode: str = "fast",
        history: list = None,
        system_prompt: str = "",
        max_reflection_loops: int = 2,
    ) -> dict:
        """
        Main entry point for chat interaction.

        Uses a Self-RAG / CRAG agentic loop:
          1. Retrieve context for the current query.
          2. Grade the retrieval — if it fails, rewrite the query and retry.
          3. Generate an answer from the (graded) context.
          4. Grade the answer for hallucinations — if it fails, rewrite and retry.
          5. Return the final answer with reflection diagnostics.

        Parameters
        ----------
        max_reflection_loops:
            Maximum number of query-rewrite + re-retrieve cycles (default 2).
        """
        reflection_loops = 0
        retrieval_grade_result = "pass"
        answer_grade_result = "grounded"

        # 1. Transform: Coreference Resolution
        history = history or []
        resolved_query = self.transformer.resolve_coreference(user_message, history)
        
        # 2. Decompose: Split compound queries
        sub_queries = self.decomposer.split_query(resolved_query)

        # Container for pooled contexts
        all_context = {
            "graph_context": [], "vector_context": [], "summary_context": [], 
            "community_context": [], "sources": [], "tools_used": [],
            "orchestrator_rationale": f"Decomposed {len(sub_queries)} queries.",
            "query_type": None
        }
        all_context_chunks = []

        # ── Retrieval → Grade → (Rewrite + Retry) loop ───────────────────
        for sq in sub_queries:
            print(f"  [Agent] Processing sub-query: {sq}")
            sq_context = self.get_context(sq)
            sq_chunks = self._collect_context_texts(sq_context)

            # Grading loop per sub-query
            for loop in range(max_reflection_loops):
                r_grade = self.reflection.grade_retrieval(sq, sq_chunks)
                print(f"  [Reflection] Retrieval grade for '{sq}' (loop {loop}): relevant={r_grade.relevant} — {r_grade.reason}")

                if r_grade.relevant:
                    break   # retrieval passed

                retrieval_grade_result = "fail"
                reflection_loops += 1

                if loop < max_reflection_loops - 1:
                    sq = self.reflection.rewrite_query(sq, r_grade.reason)
                    print(f"  [Reflection] Rewritten sub-query: {sq}")
                    sq_context = self.get_context(sq)
                    sq_chunks = self._collect_context_texts(sq_context)
                else:
                    print("  [Reflection] Max retrieval loops reached for sub-query.")

            # Merge results into pooled context
            all_context["graph_context"].extend(sq_context.get("graph_context", []))
            all_context["vector_context"].extend(sq_context.get("vector_context", []))
            all_context["summary_context"].extend(sq_context.get("summary_context", []))
            all_context["community_context"].extend(sq_context.get("community_context", []))
            
            # Deduplicate sources
            for src in sq_context.get("sources", []):
                if src not in all_context["sources"]:
                   all_context["sources"].append(src)
                   
            for t in sq_context.get("tools_used", []):
                if t not in all_context["tools_used"]:
                    all_context["tools_used"].append(t)
            
            if not all_context["query_type"]:
                all_context["query_type"] = sq_context.get("query_type")
                
            all_context_chunks.extend(sq_chunks)

        # ── Answer generation ────────────────────────────────────────────
        full_prompt = self._build_prompt(all_context, user_message, system_prompt)
        response = self.llm.complete(full_prompt)
        answer_text = response.text

        # ── Answer grounding check ───────────────────────────────────────
        a_grade = self.reflection.grade_answer(user_message, all_context_chunks, answer_text)
        print(f"  [Reflection] Answer grade: grounded={a_grade.grounded} — {a_grade.reason}")

        if not a_grade.grounded and reflection_loops < max_reflection_loops:
            answer_grade_result = "ungrounded"
            reflection_loops += 1
            print("  [Reflection] Answer ungrounded — attempting one corrective rewrite.")
            
            # Corrective retrieval directly targets the user's main requirement
            corrective_query = self.reflection.rewrite_query(
                user_message,
                failure_reason=f"The answer was ungrounded. Reason: {a_grade.reason}. Retrieve factual information to correct this."
            )
            corrective_context = self.get_context(corrective_query)
            corrective_chunks = self._collect_context_texts(corrective_context)

            # Generate new answer with corrective context
            corrective_prompt = self._build_prompt(corrective_context, user_message, system_prompt)
            corrective_response = self.llm.complete(corrective_prompt)
            
            # Final grade check
            final_a_grade = self.reflection.grade_answer(user_message, corrective_chunks, corrective_response.text)
            
            if final_a_grade.grounded:
                answer_text = corrective_response.text
                all_context = corrective_context  # Update visible logic to use the corrective sources
                answer_grade_result = "grounded"
        else:
            answer_grade_result = "grounded" if a_grade.grounded else "ungrounded"


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
        }

    def close(self):
        try:
            self.graph_store.close()
        except Exception:
            pass
