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
        """Retrieve and fuse context from all indices."""
        query_type, entity_hints = self.router.classify_query(query)
        print(f"  [Router] Query: {query_type.value}, Entities: {entity_hints}")

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
        if self.vector_index and query_type in (QueryType.LOCAL, QueryType.HYBRID):
            nodes = self.vector_index.as_retriever(similarity_top_k=5).retrieve(query)
            vector_candidates = _to_dicts(nodes)
            all_sources.extend(self.formatter.extract_sources(nodes))

        bm25_candidates = []
        if self.bm25_retriever and query_type in (QueryType.LOCAL, QueryType.HYBRID):
            nodes = self.bm25_retriever.retrieve(query)
            bm25_candidates = _to_dicts(nodes)
            all_sources.extend(self.formatter.extract_sources(nodes))

        # 2. Knowledge Graph
        graph_candidates = []
        expanded_entities = []
        if query_type in (QueryType.LOCAL, QueryType.HYBRID):
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
        if query_type in (QueryType.LOCAL, QueryType.HYBRID):
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
        if self.summary_index and query_type in (QueryType.GLOBAL, QueryType.HYBRID):
            nodes = self.summary_index.as_retriever().retrieve(query)
            summary_context = [(n.get_content(), "Summary") for n in nodes[:2]]
            all_sources.extend(self.formatter.extract_sources(nodes))

        community_context = []
        if self.community_summarizer and query_type in (QueryType.GLOBAL, QueryType.HYBRID):
            communities = self.community_summarizer.get_relevant_summaries(query, top_k=3)
            for c in communities:
                community_context.append((c.get("summary", ""), f"Community {c.get('community_id')}"))

        # 5. Entity Summarization
        summarized_graph = self.graph_service.summarize_entity_context(
            [(c["text"], c["source"]) for c in graph_candidates],
            expanded_entities
        )

        return {
            "graph_context": summarized_graph,
            "vector_context": fused_context,
            "summary_context": summary_context,
            "community_context": community_context,
            "sources": all_sources,
            "query_type": query_type,
        }

    def chat(self, user_message: str, mode: str = "fast", history: list = None, system_prompt: str = "") -> dict:
        """Main entry point for chat interaction."""
        context = self.get_context(user_message)
        
        # Build prompt sections
        prompt_parts = []
        if context["community_context"]:
            prompt_parts.append("COMMUNITY SUMMARIES:\n" + "\n".join(f"- {t}" for t, s in context["community_context"]))
        if context["graph_context"]:
            prompt_parts.append("KNOWLEDGE GRAPH:\n" + "\n".join(f"- {t}" for t, s in context["graph_context"]))
        if context["vector_context"]:
            prompt_parts.append("DOCUMENT EXCERPTS:\n" + "\n".join(f"- {t} (Source: {s})" for t, s in context["vector_context"]))

        context_str = "\n\n".join(prompt_parts) if prompt_parts else "No relevant context found."
        
        # Simplified chat logic for demonstration
        full_prompt = f"{system_prompt}\n\nContext:\n{context_str}\n\nUser: {user_message}\nAssistant:"
        
        response = self.llm.complete(full_prompt)
        
        return {
            "response": response.text,
            "sources": context["sources"],
            "query_type": context["query_type"].value,
            "suggested_prompts": self.get_suggestions(),
        }

    def close(self):
        try:
            self.graph_store.close()
        except Exception:
            pass
