import os
import json
import asyncio
import nest_asyncio
from enum import Enum
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
# Query Type Enum — used by the intelligent router
# ---------------------------------------------------------------------------
class QueryType(str, Enum):
    LOCAL = "LOCAL"    # Specific entity/fact queries
    GLOBAL = "GLOBAL"  # Thematic/summary queries across documents
    HYBRID = "HYBRID"  # Specific topic needing broader context


# ---------------------------------------------------------------------------
# Query routing keywords — fallback for when LLM router fails
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

        # ── BM25 (Sparse Lexical) Retriever ───────────────────────────
        self.bm25_retriever = None
        try:
            from llama_index.retrievers.bm25 import BM25Retriever
            bm25_dir = "./storage/bm25"
            if os.path.isdir(bm25_dir):
                self.bm25_retriever = BM25Retriever.from_persist_dir(bm25_dir)
                print("  ✓ BM25 index loaded")
            else:
                print("  ⓘ BM25 index not found (run indexer to build it)")
        except Exception as e:
            print(f"  ✗ BM25 index not available: {e}")

        print("HybridEngine ready.")

    # ──────────────────────────────────────────────────────────────────
    # Phase 4: Intelligent LLM Query Router
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_global_query_fallback(query: str) -> bool:
        """Keyword-based fallback for query classification."""
        q_lower = query.lower()
        return any(kw in q_lower for kw in _GLOBAL_QUERY_KEYWORDS)

    def _classify_query(self, query: str) -> tuple[QueryType, list[str]]:
        """
        Use the LLM to classify a query into LOCAL, GLOBAL, or HYBRID,
        and extract any entity names mentioned.

        Returns (QueryType, list_of_entity_hints).
        Falls back to keyword heuristic if the LLM call fails.
        """
        prompt = (
            "You are a query classifier for a Knowledge Graph RAG system.\n"
            "Classify the following query into exactly one category:\n\n"
            "- LOCAL: The query asks about a specific entity, fact, or relationship.\n"
            '  Example: "What is the budget for Project Alpha?"\n'
            "- GLOBAL: The query asks for themes, patterns, summaries, or comparisons across multiple topics.\n"
            '  Example: "What are the main themes across all documents?"\n'
            "- HYBRID: The query asks about a specific topic but requires broader context to answer fully.\n"
            '  Example: "Why is Project Alpha over budget compared to similar projects?"\n\n'
            "Also extract any specific entity names mentioned in the query.\n\n"
            f'Query: "{query}"\n\n'
            'Respond with ONLY a JSON object: {"type": "LOCAL|GLOBAL|HYBRID", "entities": ["entity1", "entity2"]}\n'
            "Do NOT include any explanation."
        )
        try:
            response = self.llm.complete(prompt).text.strip()
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                parsed = json.loads(response[start:end])
                qtype_str = parsed.get("type", "LOCAL").upper()
                entities = parsed.get("entities", [])
                if not isinstance(entities, list):
                    entities = []
                entities = [str(e).strip() for e in entities if e]

                if qtype_str == "GLOBAL":
                    return QueryType.GLOBAL, entities
                elif qtype_str == "HYBRID":
                    return QueryType.HYBRID, entities
                else:
                    return QueryType.LOCAL, entities
        except Exception as e:
            print(f"  [Router] LLM classification failed: {e}, using fallback")

        # Fallback to keyword heuristic
        if self._is_global_query_fallback(query):
            return QueryType.GLOBAL, []
        return QueryType.LOCAL, []

    # ──────────────────────────────────────────────────────────────────
    # Phase 1: N-Hop Graph Expansion
    # ──────────────────────────────────────────────────────────────────

    def _expand_graph_context(
        self,
        seed_entities: list[str],
        max_hops: int = 2,
        limit: int = 30,
    ) -> list[dict]:
        """
        Given a list of seed entity names, traverse 1..max_hops in Neo4j
        to discover neighboring entities and their relationships.

        Returns a list of dicts with keys:
          name, title, labels, rel_types, valid_from, valid_to, is_expanded
        """
        if not seed_entities:
            return []

        # Normalize seeds to match graph IDs (lowercase + underscores)
        normalized_seeds = [s.replace(" ", "_").lower() for s in seed_entities]

        query = f"""
        MATCH (seed)
        WHERE seed.name IN $entity_names
           OR seed.title IN $raw_names
        WITH seed
        MATCH path = (seed)-[*1..{max_hops}]-(neighbor)
        WHERE neighbor <> seed
          AND NOT any(lbl IN labels(neighbor) WHERE lbl IN ['__Community__', 'Category', 'CommunitySummary'])
          AND neighbor.name IS NOT NULL
        WITH DISTINCT neighbor,
             [r IN relationships(path) | type(r)] AS rel_types,
             [r IN relationships(path) | r.valid_from] AS valid_froms,
             [r IN relationships(path) | r.valid_to] AS valid_tos
        RETURN neighbor.name AS name,
               coalesce(neighbor.title, neighbor.name) AS title,
               labels(neighbor) AS labels,
               rel_types,
               head(valid_froms) AS valid_from,
               head(valid_tos) AS valid_to
        LIMIT $limit
        """
        try:
            results = self.graph_store.structured_query(
                query,
                param_map={
                    "entity_names": normalized_seeds,
                    "raw_names": seed_entities,
                    "limit": limit,
                },
            )
            expanded = []
            if isinstance(results, list):
                seen_names = set(normalized_seeds)
                for r in results:
                    if isinstance(r, dict):
                        name = r.get("name", "")
                    elif hasattr(r, "values"):
                        vals = list(r.values())
                        name = vals[0] if vals else ""
                        r = {
                            "name": name,
                            "title": vals[1] if len(vals) > 1 else name,
                            "labels": vals[2] if len(vals) > 2 else [],
                            "rel_types": vals[3] if len(vals) > 3 else [],
                            "valid_from": vals[4] if len(vals) > 4 else None,
                            "valid_to": vals[5] if len(vals) > 5 else None,
                        }
                    else:
                        continue

                    if name and name not in seen_names:
                        seen_names.add(name)
                        expanded.append({
                            "name": r.get("name", ""),
                            "title": r.get("title", r.get("name", "")),
                            "labels": r.get("labels", []),
                            "rel_types": r.get("rel_types", []),
                            "valid_from": r.get("valid_from"),
                            "valid_to": r.get("valid_to"),
                            "is_expanded": True,
                        })
            print(f"  [N-Hop] Expanded {len(seed_entities)} seeds → {len(expanded)} neighbors ({max_hops}-hop)")
            return expanded
        except Exception as e:
            print(f"  [N-Hop] Expansion failed: {e}")
            return []

    # ──────────────────────────────────────────────────────────────────
    # Phase 2: Entity-Centric Summarization (LLM-based)
    # ──────────────────────────────────────────────────────────────────

    def _summarize_entity_context(
        self,
        graph_context: list[tuple[str, str]],
        expanded_entities: list[dict],
    ) -> list[tuple[str, str]]:
        """
        Convert raw KG triplets + expanded entity data into natural-language
        entity profiles using the LLM.

        Returns a list of (summary_text, entity_label) tuples.
        """
        if not graph_context and not expanded_entities:
            return []

        # Build a combined profile block for the LLM
        profile_parts = []

        # From direct graph retrieval
        for text, source in graph_context:
            profile_parts.append(f"[Direct] {text} (Source: {source})")

        # From N-hop expanded entities
        for ent in expanded_entities:
            title = ent.get("title", ent.get("name", "?"))
            labels = [l for l in ent.get("labels", []) if l not in ("__Entity__", "__Node__")]
            label_str = f" ({', '.join(labels)})" if labels else ""
            rel_str = ", ".join(ent.get("rel_types", []))
            temporal = ""
            if ent.get("valid_from") or ent.get("valid_to"):
                vf = ent.get("valid_from", "?")
                vt = ent.get("valid_to", "present")
                temporal = f" [{vf} – {vt}]"
            profile_parts.append(
                f"[Related] {title}{label_str} — connected via: {rel_str}{temporal}"
            )

        if not profile_parts:
            return graph_context  # nothing to summarize

        combined = "\n".join(profile_parts[:30])  # cap to avoid exceeding context

        prompt = (
            "You are a Knowledge Graph summarizer. Given the following raw graph data "
            "(both direct facts and related/expanded context), produce a concise, "
            "readable summary grouped by entity.\n\n"
            "For each entity mentioned, write 1-3 sentences describing what it is, "
            "its key relationships, and any temporal context (dates). "
            "Clearly separate direct facts from expanded/related context.\n\n"
            f"--- GRAPH DATA ---\n{combined}\n\n"
            "Write the summary now. Keep it under 300 words. Do NOT include any preamble."
        )

        try:
            response = self.llm.complete(prompt).text.strip()
            # Return as a single summarized block
            return [(response, "Knowledge Graph (Summarized)")]
        except Exception as e:
            print(f"  [Entity Summary] LLM summarization failed: {e}")
            return graph_context  # fallback to raw context

    # ──────────────────────────────────────────────────────────────────
    # Source Extraction
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

    # ──────────────────────────────────────────────────────────────────
    # Quick Prompts / Suggestions
    # ──────────────────────────────────────────────────────────────────

    def get_suggestions(self, limit: int = 4) -> list[str]:
        """
        Dynamically suggest "quick prompts" based on the most connected entities
        or broad communities in the Knowledge Graph.
        """
        suggestions = []
        try:
            # 1. Grab top entities by relationship count (degree)
            entity_query = """
            MATCH (n)
            WHERE NOT any(lbl IN labels(n) WHERE lbl IN ['__Community__', 'Category', 'CommunitySummary'])
              AND n.name IS NOT NULL
            WITH n, size((n)--()) AS degree
            ORDER BY degree DESC
            LIMIT 5
            RETURN n.name AS name, degree
            """
            entities = self.graph_store.structured_query(entity_query)
            if isinstance(entities, list):
                for e in entities:
                    if isinstance(e, dict) and e.get("name"):
                        suggestions.append(f"Tell me about {e['name']}")
                    elif hasattr(e, "values"):
                        vals = list(e.values())
                        if vals and vals[0]:
                            suggestions.append(f"Tell me about {vals[0]}")

            # 2. Grab top communities (themes) if available
            if self.community_summarizer:
                summaries = self.community_summarizer.get_all_summaries()
                for comm in summaries[:3]:
                    keys = comm.get("key_entities", "")
                    if keys:
                        parts = keys.split(",")
                        if parts:
                            top_entity = parts[0].strip()
                            suggestions.append(f"What is the connection between {top_entity} and others?")

            # Deduplicate and limit
            seen = set()
            unique_suggestions = []
            for s in suggestions:
                if s not in seen:
                    seen.add(s)
                    unique_suggestions.append(s)
                    if len(unique_suggestions) >= limit:
                        break

            return unique_suggestions

        except Exception as e:
            print(f"  [Suggestions] Failed to get suggestions: {e}")
            return [
                "What forms of renewable energy are discussed?",
                "Tell me about recent tech advancements",
                "Summarize the key themes in the documents",
            ]


    # ──────────────────────────────────────────────────────────────────
    # Hybrid Retrieval with RRF
    # ──────────────────────────────────────────────────────────────────

    def _reciprocal_rank_fusion(self, list_of_ranked_lists, k=60) -> list:
        """
        Fuses multiple ranked lists using Reciprocal Rank Fusion (RRF).
        list_of_ranked_lists is a list of lists of dictionaries. 
        Each dictionary must have a unique 'id' or 'text' to identify the item.
        """
        rrf_scores = {}
        item_map = {}
        
        for ranked_list in list_of_ranked_lists:
            for rank, item in enumerate(ranked_list):
                # Use 'id' if available, otherwise hash the text/title for deduplication
                item_id = item.get("id", hash(item.get("text", "")))
                
                if item_id not in rrf_scores:
                    rrf_scores[item_id] = 0.0
                    item_map[item_id] = item
                    
                # RRF score formula: 1 / (k + rank)
                rrf_scores[item_id] += 1.0 / (k + rank + 1)
                
        # Sort by RRF score descending
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [item_map[item_id] for item_id, _ in sorted_items]

    def _hybrid_graph_traversal(self, seed_chunks: list[dict], max_nodes: int = 10) -> dict:
        """
        Uses the new Graph structure (MENTIONS & SIMILAR_TO) to find related context
        starting from vector-retrieved seed chunks.
        """
        if not seed_chunks or not self.graph_store:
            return {"chunks": [], "entities": []}

        # Extract text/IDs from seeds
        seed_texts = [s.get("text", "") for s in seed_chunks if s.get("text")]
        
        query = """
        UNWIND $seed_texts AS text
        MATCH (seed:__Node__) WHERE seed.text = text
        
        // 1. Get entities mentioned in these chunks
        OPTIONAL MATCH (seed)-[:MENTIONS]->(e:__Entity__)
        
        // 2. Get semantically similar chunks
        OPTIONAL MATCH (seed)-[sim:SIMILAR_TO]->(neighbor_chunk:__Node__)
        
        RETURN 
            collect(DISTINCT {name: e.name, type: labels(e)[0], title: e.title}) AS entities,
            collect(DISTINCT {text: neighbor_chunk.text, score: sim.score}) AS neighbor_chunks
        """
        
        try:
            results = self.graph_store.structured_query(query, param_map={"seed_texts": seed_texts})
            
            entities = []
            neighbor_chunks = []
            
            if isinstance(results, list) and results:
                record = results[0]
                
                # Parse entities
                raw_entities = record.get("entities", [])
                if hasattr(record, "values") and not isinstance(record, dict):
                    raw_entities = record.values()[0] if len(record.values()) > 0 else []
                    
                for e in raw_entities:
                    if isinstance(e, dict) and e.get("name"):
                        # Format as expanded entities for _summarize_entity_context
                        entities.append({
                            "name": e.get("name"),
                            "title": e.get("title", e.get("name")),
                            "labels": [e.get("type", "Entity")],
                            "rel_types": ["MENTIONS"],
                            "is_expanded": True
                        })
                        
                # Parse neighbor chunks
                raw_neighbors = record.get("neighbor_chunks", [])
                if hasattr(record, "values") and not isinstance(record, dict):
                    raw_neighbors = record.values()[1] if len(record.values()) > 1 else []
                    
                for nc in raw_neighbors:
                    if isinstance(nc, dict) and nc.get("text"):
                        neighbor_chunks.append({
                            "text": nc["text"],
                            "source_title": "Semantic Neighbor"
                        })

            return {
                "chunks": neighbor_chunks[:max_nodes], 
                "entities": entities[:max_nodes*2]
            }
        except Exception as e:
            print(f"  [Hybrid Traversal] Failed: {e}")
            return {"chunks": [], "entities": []}


    def get_context(self, query: str) -> dict:
        """
        Retrieve context from all available indexes, using RRF for fusion and 
        hybrid graph traversal to expand context.
        """
        # Step 0: Classify the query
        query_type, entity_hints = self._classify_query(query)
        print(f"  [Router] Query classified as {query_type.value}, entities: {entity_hints}")

        all_sources = []

        def _get_context_with_sources(nodes):
            results = []
            for node in nodes:
                meta = getattr(node, "metadata", {})
                if not meta and hasattr(node, "node"):
                    meta = getattr(node.node, "metadata", {})
                title = meta.get("title", meta.get("file_name", ""))
                file_path = meta.get("file_path", "")
                if not title and file_path:
                    title = os.path.basename(file_path).rsplit(".", 1)[0]
                results.append({"text": node.text, "source": title, "metadata": meta})
            return results

        # ── 1. Vector retrieval (Semantic Seeds) ──────────────────────
        vector_candidates = []
        if self.vector_index and query_type in (QueryType.LOCAL, QueryType.HYBRID):
            try:
                vector_nodes = self.vector_index.as_retriever(similarity_top_k=10).retrieve(query)
                vector_candidates = _get_context_with_sources(vector_nodes)
                all_sources.extend(self._extract_sources(vector_nodes))
            except Exception as e:
                print(f"  Vector retrieval error: {e}")

        # ── 1b. BM25 retrieval (Lexical Seeds) ────────────────────────
        bm25_candidates = []
        if self.bm25_retriever and query_type in (QueryType.LOCAL, QueryType.HYBRID):
            try:
                bm25_nodes = self.bm25_retriever.retrieve(query)
                bm25_candidates = _get_context_with_sources(bm25_nodes)
                all_sources.extend(self._extract_sources(bm25_nodes))
                print(f"  [BM25] Retrieved {len(bm25_candidates)} candidates")
            except Exception as e:
                print(f"  BM25 retrieval error: {e}")

        # ── 2. Knowledge Graph text retrieval ──────────────────────────
        graph_candidates = []
        expanded_entities = []
        
        if query_type in (QueryType.LOCAL, QueryType.HYBRID):
            try:
                kg_retriever = self.kg_index.as_retriever(include_text=True, similarity_top_k=10)
                graph_nodes = kg_retriever.retrieve(query)
                graph_candidates = _get_context_with_sources(graph_nodes)
                all_sources.extend(self._extract_sources(graph_nodes))
                
                # Fetch N-hop for entity hints
                kg_entity_names = []
                for gn in graph_nodes:
                    meta = getattr(gn, "metadata", getattr(getattr(gn, "node", None), "metadata", {}))
                    if meta.get("name"):
                        kg_entity_names.append(meta.get("name"))
                
                all_seeds = list(set(entity_hints + kg_entity_names))
                if all_seeds:
                    expanded_entities.extend(self._expand_graph_context(all_seeds, max_hops=1, limit=15))
            except Exception as e:
                print(f"  KG retrieval error: {e}")
                
        elif query_type == QueryType.GLOBAL and entity_hints:
            expanded_entities = self._expand_graph_context(entity_hints, max_hops=1, limit=15)

        # ── 3. Tri-Fusion via RRF & Hybrid Graph Traversal ────────────
        vector_context_fused = []
        
        if query_type in (QueryType.LOCAL, QueryType.HYBRID):
            # Fuse Vector + BM25 + Graph-text retrieval via RRF
            fusion_lists = [l for l in [vector_candidates, bm25_candidates, graph_candidates] if l]
            fused_chunks = self._reciprocal_rank_fusion(fusion_lists) if fusion_lists else []
            
            # Take top 5 for final prompt
            top_fused = fused_chunks[:5]
            vector_context_fused = [(c["text"], c["source"]) for c in top_fused]
            
            # Hybrid Graph Traversal from top seeds
            traversal_results = self._hybrid_graph_traversal(top_fused, max_nodes=3)
            
            # Add semantic neighbors as context
            for nc in traversal_results["chunks"]:
                vector_context_fused.append((nc["text"], nc["source_title"]))
                
            # Add connected entities to our expanded list
            expanded_entities.extend(traversal_results["entities"])

        # ── 4. Summary retrieval ──────────────────────────────────────
        summary_context = []
        if self.summary_index and query_type in (QueryType.GLOBAL, QueryType.HYBRID):
            try:
                summary_nodes = self.summary_index.as_retriever().retrieve(query)
                summary_formatted = _get_context_with_sources(summary_nodes)
                summary_context = [(c["text"], c["source"]) for c in summary_formatted[:3]]
                all_sources.extend(self._extract_sources(summary_nodes))
            except Exception as e:
                print(f"  Summary retrieval error: {e}")

        # ── 5. Community retrieval (GraphRAG global search) ───────────
        community_context = []
        if self.community_summarizer and query_type in (QueryType.GLOBAL, QueryType.HYBRID):
            try:
                top_k = 5 if query_type == QueryType.GLOBAL else 2
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

        # ── 6. Entity-Centric Summarization ───────────────────────────
        summarized_graph = []
        raw_graph_tuples = [(c["text"], c["source"]) for c in graph_candidates]
        
        if raw_graph_tuples or expanded_entities:
            # Deduplicate expanded entities
            seen_ent = set()
            uniq_ent = []
            for e in expanded_entities:
                key = e.get("name", "")
                if key and key not in seen_ent:
                    seen_ent.add(key)
                    uniq_ent.append(e)
                    
            summarized_graph = self._summarize_entity_context(
                raw_graph_tuples[:5], uniq_ent
            )

        # Deduplicate sources
        seen = set()
        unique_sources = []
        for s in all_sources:
            key = f"{s['title']}|{s['category']}"
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)

        return {
            "graph_context": summarized_graph,
            "raw_graph_context": raw_graph_tuples,
            "vector_context": vector_context_fused,
            "summary_context": summary_context,
            "community_context": community_context,
            "sources": unique_sources,
            "query_type": query_type,
        }

    # ──────────────────────────────────────────────────────────────────
    # Chat
    # ──────────────────────────────────────────────────────────────────

    # ── Planning mode context window (larger than fast)
    PLANNING_CONTEXT_WINDOW = 16384
    FAST_CONTEXT_WINDOW = int(os.getenv("OLLAMA_CONTEXT_WINDOW", "8192"))

    def chat(self, user_message: str, mode: str = "fast", history: list = None, system_prompt: str = "") -> dict:
        """
        End-to-end RAG chat:
          1. Classify query via LLM router (LOCAL / GLOBAL / HYBRID)
          2. Retrieve context from KG + N-hop + vector + summary + communities
          3. Summarize entity profiles (LLM-based)
          4. Build augmented prompt (routed by query type)
          5. Call LLM
          6. Parse response to extract citations
          7. Return filtered answer + relevant sources
        """
        context_data = self.get_context(user_message)
        sources = context_data["sources"]
        query_type = context_data.get("query_type", QueryType.LOCAL)

        # Build combined context string with source tag labeling
        context_parts = []

        # Helper to format context blocks with labels
        def format_blocks(label, blocks):
            formatted = []
            for text, source in blocks:
                src_label = f" (Source: {source})" if source else ""
                formatted.append(f"- {text}{src_label}")
            return f"{label}:\n" + "\n".join(formatted) if formatted else ""

        # For GLOBAL queries, prioritize community context at the top
        if query_type == QueryType.GLOBAL and context_data["community_context"]:
            context_parts.append(format_blocks(
                "THEMATIC COMMUNITY SUMMARIES (use these for big-picture analysis)",
                context_data["community_context"]
            ))

        if context_data["graph_context"]:
            context_parts.append(format_blocks(
                "KNOWLEDGE GRAPH DATA (entity profiles with relationships)",
                context_data["graph_context"][:5]
            ))

        if context_data["vector_context"]:
            context_parts.append(format_blocks("DOCUMENT EXCERPTS", context_data["vector_context"][:5]))

        if context_data["summary_context"]:
            context_parts.append(format_blocks("DOCUMENT SUMMARIES", context_data["summary_context"][:3]))

        # For HYBRID, append community context as supplementary
        if query_type == QueryType.HYBRID and context_data["community_context"]:
            context_parts.append(format_blocks(
                "THEMATIC CONTEXT (supplementary)",
                context_data["community_context"][:2]
            ))

        history_text = ""
        if history:
            history_blocks = []
            for msg in history[-6:]:
                role = msg.get("role", "user").upper()
                content = msg.get("content", "").strip()
                if content:
                    history_blocks.append(f"{role}: {content}")
            if history_blocks:
                history_text = "CONVERSATION HISTORY (Previous turns):\n" + "\n\n".join(history_blocks) + "\n\n---\n\n"

        has_context = bool(context_parts)
        if not has_context:
            suggs = self.get_suggestions(limit=3)
            sg_str = "\n".join(f"- {s}" for s in suggs)
            context_parts.append(
                f"NO RELEVANT CONTEXT FOUND.\n"
                f"The user's query did not match any documents in the knowledge base.\n"
                f"Please politely inform the user that you don't have specific documents on this topic, but suggest they ask about the following topics that ARE in the knowledge base:\n"
                f"{sg_str}"
            )

        combined_context = "\n\n---\n\n".join(filter(None, context_parts)) if context_parts else "No relevant context found."
        combined_context = history_text + combined_context

        # Query-type-specific instructions
        if query_type == QueryType.LOCAL:
            focus_instruction = "Focus on the specific facts and entity relationships provided below."
        elif query_type == QueryType.GLOBAL:
            focus_instruction = "Synthesize the thematic summaries and document overviews below to provide a comprehensive, big-picture answer."
        else:  # HYBRID
            focus_instruction = "Use both the specific entity data and the broader thematic context below to provide a thorough answer."

        custom_system_context = f"\n\nADDITIONAL SYSTEM INSTRUCTIONS:\n{system_prompt}\n" if system_prompt and system_prompt.strip() else ""

        is_planning = (mode == "planning")

        if is_planning:
            # Planning mode: chain-of-thought reasoning, larger context
            num_ctx = self.PLANNING_CONTEXT_WINDOW
            prompt = f"""You are a meticulous AI research assistant operating in PLANNING MODE.
{focus_instruction}{custom_system_context}

Your task is to reason carefully and thoroughly before giving a final answer.
Use the provided context to inform your answer if it is relevant. If the context is insufficient, or if the user is asking a general question, you may use your broad knowledge to answer. If the request is ambiguous or you need more specific context from the user, you may ask clarifying questions.

{combined_context}

USER QUESTION: {user_message}

INSTRUCTIONS — PLANNING MODE:
1. First, write a brief <thinking> section where you:
   a. Identify what the user is asking
   b. Note which context blocks are directly relevant
   c. Identify any gaps or ambiguities that require asking the user
2. Then write your final answer (or follow-up questions) in a <answer> section:
   - Be thorough and well-structured (use bullet points or sections where helpful)
   - Include temporal context (dates, time ranges) when available
   - Cite the sources inline where possible
   - If you need clarification from the user, ask them directly.
3. At the very end list: [SOURCES_USED]: SourceTitle1, SourceTitle2, ...
   If no sources were used, write [SOURCES_USED]: None
4. After sources, provide exactly 3 suggested follow-up questions for the user based on your response.
   CRITICAL: Use the exact format [FOLLOW_UP]: Question 1 | Question 2 | Question 3 (separator is the pipe character '|').
"""
        else:
            # Fast mode: direct, concise answer
            num_ctx = self.FAST_CONTEXT_WINDOW
            prompt = f"""You are a helpful AI assistant. {focus_instruction}{custom_system_context}
Use the provided context to inform your answer if it is relevant. If the context is insufficient, or if the user is asking a general question, use your general knowledge. If you need more clarification from the user, ask them follow-up questions.

{combined_context}

USER QUESTION: {user_message}

CRITICAL INSTRUCTIONS:
1. Provide a clear, well-structured answer, or ask clarifying questions if needed.
2. Include temporal context (dates, time ranges) when available in the data.
3. At the very end of your response, list the unique source titles you actually used to form your answer.
4. Use the exact format: [SOURCES_USED]: SourceTitle1, SourceTitle2, ...
5. If you cannot answer the question, didn't use any context, or are asking a clarifying question, say "[SOURCES_USED]: None".
6. After sources, provide exactly 3 suggested follow-up questions for the user based on your response.
   CRITICAL: Use the exact format [FOLLOW_UP]: Question 1 | Question 2 | Question 3 (separator is the pipe character '|').
"""

        # Call LLM (override num_ctx per-request for planning mode)
        call_kwargs = {}
        if is_planning:
            call_kwargs = {"additional_kwargs": {"num_ctx": num_ctx}}
        response_obj = self.llm.complete(prompt, **call_kwargs)
        raw_response = str(response_obj)

        # Extract performance metrics
        context_window = num_ctx if is_planning else self.FAST_CONTEXT_WINDOW
        stats = {
            "context_window": context_window,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "context_utilization": 0.0,
            "tps": 0.0,
        }
        if hasattr(response_obj, "raw") and isinstance(response_obj.raw, dict):
            raw = response_obj.raw
            eval_count = raw.get("eval_count", 0)
            eval_duration = raw.get("eval_duration", 0)
            prompt_eval_count = raw.get("prompt_eval_count", 0)

            stats["prompt_tokens"] = prompt_eval_count
            stats["completion_tokens"] = eval_count
            stats["total_tokens"] = prompt_eval_count + eval_count

            if eval_count > 0 and eval_duration > 0:
                tps = eval_count / (eval_duration / 1e9)
                stats["tps"] = round(tps, 2)

            total_tokens = prompt_eval_count + eval_count
            if context_window > 0:
                utilization = total_tokens / context_window
                stats["context_utilization"] = round(utilization, 4)

        # Parse the response for citations and follow-ups
        import re
        answer = raw_response
        cited_sources = []
        suggested_prompts = []

        # Split on FOLLOW_UP marker (handles optional brackets)
        follow_parts = re.split(r'\[?FOLLOW_UP\]?:', raw_response, flags=re.IGNORECASE)
        if len(follow_parts) > 1:
            raw_response = follow_parts[0].strip()
            followup_str = follow_parts[1].strip()
            if followup_str:
                # Try splitting by pipe first
                if "|" in followup_str:
                    suggested_prompts = [q.strip() for q in followup_str.split("|") if q.strip()]
                else:
                    # Fallback: split by question mark followed by space or newline
                    suggested_prompts = [q.strip() + "?" for q in re.split(r'\?\s*', followup_str) if q.strip()]
                    suggested_prompts = [q.replace("??", "?") for q in suggested_prompts]
                
                # Limit to 3
                suggested_prompts = suggested_prompts[:3]

        # Split on SOURCES_USED marker (handles optional brackets)
        source_parts = re.split(r'\[?SOURCES_USED\]?:', raw_response, flags=re.IGNORECASE)
        if len(source_parts) > 1:
            answer = source_parts[0].strip()
            source_list_str = source_parts[1].strip()

            if source_list_str and source_list_str.lower() != "none":
                titles = [t.strip() for t in source_list_str.split(",") if t.strip()]
                cited_sources = titles
        else:
            answer = raw_response.strip()

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
            "query_type": query_type.value,
            "graph_context": context_data.get("raw_graph_context", context_data.get("graph_context", [])),
            "suggested_prompts": suggested_prompts,
        }

    def close(self):
        """Clean up connections."""
        try:
            self.graph_store.close()
        except Exception:
            pass
        Settings.llm = None
        Settings.embed_model = None

