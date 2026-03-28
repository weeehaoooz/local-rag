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

# ---------------------------------------------------------------------------
# Query routing keywords — used to detect global/thematic queries
# ---------------------------------------------------------------------------
_GLOBAL_QUERY_KEYWORDS = [
    "main theme", "overall", "summarize", "summary", "overview",
    "big picture", "across all", "general", "common pattern",
    "recurring", "high level", "key takeaway", "in general",
    "holistic", "what are the themes", "across documents",
]


class HybridEngine:
    """
    KG-RAG query engine that uses:
      1. PropertyGraphIndex (Neo4j KG as retrieval index)
      2. VectorStoreIndex (semantic search)
      3. SummaryIndex (structural/overview retrieval)
      4. CommunityRetriever (GraphRAG-style global/thematic retrieval)

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
        self.kg_index = PropertyGraphIndex.from_existing(
            property_graph_store=self.graph_store,
            llm=self.llm,
            embed_model=self.embed_model,
        )

        # ── Community Retriever (GraphRAG) ────────────────────────────
        try:
            from indexers.community import CommunitySummarizer
            self.community_summarizer = CommunitySummarizer(self.graph_store, llm=self.llm)
            # Quick check: are there any community summaries?
            test = self.community_summarizer.get_all_summaries()
            if test:
                print(f"  ✓ Community index loaded ({len(test)} communities)")
            else:
                print("  ⓘ No community summaries found (run: python scripts/detect_communities.py --summarize)")
        except Exception as e:
            print(f"  ✗ Community retriever not available: {e}")
            self.community_summarizer = None

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
    # Query Routing
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_global_query(query: str) -> bool:
        """
        Heuristic to detect whether a query is global/thematic (suited for
        community-level retrieval) vs. local/factual (suited for KG + vector).
        """
        q_lower = query.lower()
        return any(kw in q_lower for kw in _GLOBAL_QUERY_KEYWORDS)

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
        Returns context texts (now as lists of (text, source_title) tuples) and combined source list.
        """
        all_sources = []
        is_global = self._is_global_query(query)
        
        # Helper to get (text, source_title) from nodes
        def _get_context_with_sources(nodes):
            results = []
            for node in nodes:
                meta = node.metadata if hasattr(node, "metadata") else {}
                if not meta and hasattr(node, "node"):
                    meta = getattr(node.node, "metadata", {})
                
                title = meta.get("title", meta.get("file_name", ""))
                file_path = meta.get("file_path", "")
                if not title and file_path:
                    title = os.path.basename(file_path).rsplit(".", 1)[0]
                
                results.append((node.text, title))
            return results

        # 1. Knowledge Graph retrieval
        graph_context = []
        try:
            kg_retriever = self.kg_index.as_retriever(include_text=True, similarity_top_k=5)
            graph_nodes = kg_retriever.retrieve(query)
            graph_context = _get_context_with_sources(graph_nodes)
            all_sources.extend(self._extract_sources(graph_nodes))
        except Exception as e:
            print(f"  KG retrieval error: {e}")

        # 2. Vector retrieval
        vector_context = []
        if self.vector_index:
            try:
                vector_nodes = self.vector_index.as_retriever(similarity_top_k=5).retrieve(query)
                vector_context = _get_context_with_sources(vector_nodes)
                all_sources.extend(self._extract_sources(vector_nodes))
            except Exception as e:
                print(f"  Vector retrieval error: {e}")

        # 3. Summary retrieval
        summary_context = []
        if self.summary_index:
            try:
                summary_nodes = self.summary_index.as_retriever().retrieve(query)
                summary_context = _get_context_with_sources(summary_nodes)
                all_sources.extend(self._extract_sources(summary_nodes))
            except Exception as e:
                print(f"  Summary retrieval error: {e}")

        # 4. Community retrieval (GraphRAG global search)
        community_context = []
        if self.community_summarizer:
            try:
                # For global queries, fetch more communities; for local, fewer
                top_k = 5 if is_global else 2
                relevant_communities = self.community_summarizer.get_relevant_summaries(
                    query, top_k=top_k
                )
                for comm in relevant_communities:
                    summary = comm.get("summary", "")
                    key_entities = comm.get("key_entities", "")
                    cid = comm.get("community_id", "?")
                    if summary:
                        label = f"Community {cid} ({key_entities})"
                        community_context.append((summary, label))
            except Exception as e:
                print(f"  Community retrieval error: {e}")

        # Deduplicate sources
        seen = set()
        unique_sources = []
        for s in all_sources:
            key = f"{s['title']}|{s['category']}"
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)

        return {
            "graph_context": graph_context,
            "vector_context": vector_context,
            "summary_context": summary_context,
            "community_context": community_context,
            "sources": unique_sources,
            "is_global": is_global,
        }

    # ──────────────────────────────────────────────────────────────────
    # Chat
    # ──────────────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> dict:
        """
        End-to-end RAG chat:
          1. Retrieve context from KG + vector + summary + communities
          2. Build augmented prompt (route: local vs global)
          3. Call LLM
          4. Parse response to extract citations
          5. Return filtered answer + relevant sources
        """
        context_data = self.get_context(user_message)
        sources = context_data["sources"]
        is_global = context_data.get("is_global", False)

        # Build combined context string with source tag labeling
        context_parts = []
        
        # Helper to format context blocks with labels
        def format_blocks(label, blocks):
            formatted = []
            for text, source in blocks:
                # Add source label to each block if title is present
                src_label = f" (Source: {source})" if source else ""
                formatted.append(f"- {text}{src_label}")
            return f"{label}:\n" + "\n".join(formatted) if formatted else ""

        # For global queries, prioritize community context
        if is_global and context_data["community_context"]:
            context_parts.append(format_blocks(
                "THEMATIC COMMUNITY SUMMARIES (use these for big-picture analysis)",
                context_data["community_context"]
            ))

        if context_data["graph_context"]:
            context_parts.append(format_blocks("KNOWLEDGE GRAPH DATA", context_data["graph_context"][:5]))

        if context_data["vector_context"]:
            context_parts.append(format_blocks("DOCUMENT EXCERPTS", context_data["vector_context"][:5]))

        if context_data["summary_context"]:
            context_parts.append(format_blocks("DOCUMENT SUMMARIES", context_data["summary_context"][:3]))

        # For non-global queries, append community context at the end (supplementary)
        if not is_global and context_data["community_context"]:
            context_parts.append(format_blocks(
                "THEMATIC CONTEXT (supplementary)",
                context_data["community_context"][:2]
            ))

        combined_context = "\n\n---\n\n".join(filter(None, context_parts)) if context_parts else "No relevant context found."

        prompt = f"""You are a helpful AI assistant. Use the following retrieved information to answer the user's question accurately and comprehensively.
Base your answer ONLY on the provided context. If the context doesn't contain enough information to answer fully, say so.

{combined_context}

USER QUESTION: {user_message}

CRITICAL INSTRUCTIONS:
1. Provide a clear, well-structured answer.
2. At the very end of your response, list the unique source titles you actually used to form your answer.
3. Use the exact format: [SOURCES_USED]: SourceTitle1, SourceTitle2, ...
4. If you cannot answer the question or didn't use any context, do not list any sources or say "[SOURCES_USED]: None".
"""

        # Call LLM
        response_obj = self.llm.complete(prompt)
        raw_response = str(response_obj)
        
        # Extract performance metrics
        stats = {}
        if hasattr(response_obj, "raw") and isinstance(response_obj.raw, dict):
            raw = response_obj.raw
            eval_count = raw.get("eval_count", 0)
            eval_duration = raw.get("eval_duration", 0)
            prompt_eval_count = raw.get("prompt_eval_count", 0)
            
            if eval_count > 0 and eval_duration > 0:
                # Convert nanoseconds to seconds
                tps = eval_count / (eval_duration / 1e9)
                stats["tps"] = round(tps, 2)
            
            # Context utilization
            total_tokens = prompt_eval_count + eval_count
            context_window = getattr(self.llm, "context_window", 8192)
            if context_window > 0:
                utilization = total_tokens / context_window
                stats["context_utilization"] = round(utilization, 4)

        # Parse the response for citations
        answer = raw_response
        cited_sources = []
        
        marker = "[SOURCES_USED]:"
        if marker in raw_response:
            parts = raw_response.split(marker)
            answer = parts[0].strip()
            source_list_str = parts[1].strip()
            
            if source_list_str and source_list_str.lower() != "none":
                # Clean up and split titles
                titles = [t.strip() for t in source_list_str.split(",") if t.strip()]
                cited_sources = titles

        # Filter the original source list based on what the LLM cited
        filtered_sources = []
        lowercase_cited = [c.lower() for c in cited_sources]
        
        for s in sources:
            if s["title"].lower() in lowercase_cited:
                filtered_sources.append(s)
            elif any(s["title"].lower() in c.lower() or c.lower() in s["title"].lower() for c in cited_sources):
                 filtered_sources.append(s)

        return {
            "response": answer,
            "sources": filtered_sources,
            "stats": stats,
            "graph_context": context_data.get("graph_context", []),
        }

    def close(self):
        """Clean up connections."""
        try:
            self.graph_store.close()
        except Exception:
            pass
        Settings.llm = None
        Settings.embed_model = None

