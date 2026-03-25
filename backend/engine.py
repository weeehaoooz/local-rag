import os
import asyncio
import nest_asyncio
from dotenv import load_dotenv
from llama_index.core import (
    PropertyGraphIndex, VectorStoreIndex, SummaryIndex,
    StorageContext, Settings, load_index_from_storage,
)
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Skip nest_asyncio when running under Uvicorn/FastAPI with uvloop
# nest_asyncio.apply()

# Load environment variables
load_dotenv()


class HybridEngine:
    """
    KG-RAG query engine that uses:
      1. PropertyGraphIndex (Neo4j KG as retrieval index)
      2. VectorStoreIndex (semantic search)
      3. SummaryIndex (structural/overview retrieval)

    Each retrieval path returns source metadata so the caller
    can attribute answers to original documents.
    """

    def __init__(self):
        # We don't need to force an event loop or nest_asyncio here
        # since FastAPI/Uvicorn already provides and manages an async loop.
        
        # ── LLM & Embedding ───────────────────────────────────────────
        self.llm = Ollama(
            model="llama3:latest",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            request_timeout=720.0,
            context_window=int(os.getenv("OLLAMA_CONTEXT_WINDOW", "8192")),
            additional_kwargs={
                "num_ctx": int(os.getenv("OLLAMA_CONTEXT_WINDOW", "8192")),
            },
        )
        self.embed_model = OllamaEmbedding(
            model_name="nomic-embed-text",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            request_timeout=360.0,
        )

        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        # ── Neo4j Property Graph Store ────────────────────────────────
        self.graph_store = Neo4jPropertyGraphStore(
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),
            url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            refresh_schema=False,
        )

        # ── PropertyGraphIndex (KG retriever) ─────────────────────────
        self.kg_index = PropertyGraphIndex(
            nodes=[],
            property_graph_store=self.graph_store,
            llm=self.llm,
            embed_model=self.embed_model,
            use_async=False,
        )

        # ── Vector & Summary indexes from persisted storage ───────────
        print("Loading indexes from ./storage...")
        try:
            vector_ctx = StorageContext.from_defaults(persist_dir="./storage/vector")
            self.vector_index = load_index_from_storage(vector_ctx)
            print("  ✓ Vector index loaded")
        except Exception as e:
            print(f"  ✗ Vector index not available: {e}")
            self.vector_index = None

        try:
            summary_ctx = StorageContext.from_defaults(persist_dir="./storage/summary")
            self.summary_index = load_index_from_storage(summary_ctx)
            print("  ✓ Summary index loaded")
        except Exception as e:
            print(f"  ✗ Summary index not available: {e}")
            self.summary_index = None

        print("HybridEngine ready.")

    # ──────────────────────────────────────────────────────────────────
    # Retrieval
    # ──────────────────────────────────────────────────────────────────

    def _extract_sources(self, nodes) -> list[dict]:
        """Extract unique source metadata from retrieved nodes."""
        seen = set()
        sources = []
        for node in nodes:
            meta = node.metadata if hasattr(node, "metadata") else {}
            # Try node.node.metadata for NodeWithScore wrappers
            if not meta and hasattr(node, "node"):
                meta = getattr(node.node, "metadata", {})

            title = meta.get("title", meta.get("file_name", ""))
            category = meta.get("category", "")
            file_path = meta.get("file_path", "")

            if not title and file_path:
                title = os.path.basename(file_path).rsplit(".", 1)[0]

            key = f"{title}|{category}"
            if title and key not in seen:
                seen.add(key)
                sources.append({
                    "title": title,
                    "category": category,
                    "file": os.path.basename(file_path) if file_path else "",
                })
        return sources

    def get_context(self, query: str) -> dict:
        """
        Retrieve context from all available indexes.
        Returns context texts and combined source list.
        """
        all_sources = []

        # 1. Knowledge Graph retrieval (KG as index)
        graph_nodes = []
        try:
            kg_retriever = self.kg_index.as_retriever(
                include_text=True,
                similarity_top_k=5,
            )
            graph_nodes = kg_retriever.retrieve(query)
            all_sources.extend(self._extract_sources(graph_nodes))
        except Exception as e:
            print(f"  KG retrieval error: {e}")

        # 2. Vector retrieval (semantic search)
        vector_nodes = []
        if self.vector_index:
            try:
                vector_nodes = self.vector_index.as_retriever(
                    similarity_top_k=5,
                ).retrieve(query)
                all_sources.extend(self._extract_sources(vector_nodes))
            except Exception as e:
                print(f"  Vector retrieval error: {e}")

        # 3. Summary retrieval
        summary_nodes = []
        if self.summary_index:
            try:
                summary_nodes = self.summary_index.as_retriever().retrieve(query)
                all_sources.extend(self._extract_sources(summary_nodes))
            except Exception as e:
                print(f"  Summary retrieval error: {e}")

        # Deduplicate sources
        seen = set()
        unique_sources = []
        for s in all_sources:
            key = f"{s['title']}|{s['category']}"
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)

        return {
            "graph_context": [n.text for n in graph_nodes],
            "vector_context": [n.text for n in vector_nodes],
            "summary_context": [n.text for n in summary_nodes],
            "sources": unique_sources,
        }

    # ──────────────────────────────────────────────────────────────────
    # Chat
    # ──────────────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> dict:
        """
        End-to-end RAG chat:
          1. Retrieve context from KG + vector + summary
          2. Build augmented prompt
          3. Call LLM
          4. Return answer + sources
        """
        context = self.get_context(user_message)

        # Build combined context string
        context_parts = []

        if context["graph_context"]:
            kg_text = "\n".join(context["graph_context"][:5])
            context_parts.append(f"KNOWLEDGE GRAPH DATA:\n{kg_text}")

        if context["vector_context"]:
            vec_text = "\n".join(context["vector_context"][:5])
            context_parts.append(f"DOCUMENT EXCERPTS:\n{vec_text}")

        if context["summary_context"]:
            sum_text = "\n".join(context["summary_context"][:3])
            context_parts.append(f"DOCUMENT SUMMARIES:\n{sum_text}")

        combined_context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant context found."

        prompt = f"""You are a helpful AI assistant. Use the following retrieved information to answer the user's question accurately and comprehensively. Base your answer ONLY on the provided context. If the context doesn't contain enough information to answer fully, say so.

{combined_context}

USER QUESTION: {user_message}

Provide a clear, well-structured answer. If information comes from multiple sources, synthesize it coherently."""

        # Call LLM
        response = self.llm.complete(prompt)

        return {
            "response": str(response),
            "sources": context["sources"],
        }

    def close(self):
        """Clean up connections."""
        try:
            self.graph_store.close()
        except Exception:
            pass
        Settings.llm = None
        Settings.embed_model = None
