from llama_index.llms.ollama import Ollama
from datetime import datetime, timezone
import os
import re
import asyncio
import difflib
import nest_asyncio
nest_asyncio.apply()
import numpy as np
from typing import List, Any, Optional, Dict, Tuple

from llama_index.core import PropertyGraphIndex, StorageContext, Settings
from llama_index.core.schema import Document, TextNode, NodeRelationship
from llama_index.core.indices.property_graph import (
    SchemaLLMPathExtractor,
    SimpleLLMPathExtractor,
    ImplicitPathExtractor,
)

from llama_index.core.node_parser import SentenceSplitter
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from indexing.base import BaseIndexer


from indexing.graph_extractor import (
    RobustSchemaExtractor,
    _small_to_big_parse,
    _build_extractors_from_guardrails,
)
from indexing.graph_cleaner import GraphCleaner
from indexing.community import (
    build_networkx_graph,
    detect_communities,
    write_communities_to_neo4j,
    summarize_communities,
    CommunitySummarizer,
)

class GraphIndexer(BaseIndexer):
    """Indexer for Property Graph (Neo4j) using PropertyGraphIndex."""

    def __init__(self, storage_context: StorageContext):
        super().__init__("graph")
        self.storage_context = storage_context
        self.index: Optional[PropertyGraphIndex] = None
        self._entity_cache: Dict[str, List[str]] = {}

    def clear_cache(self):
        """Clear the entity cache."""
        self._entity_cache = {}

    # ------------------------------------------------------------------
    # PUBLIC: Synchronous wrapper
    # ------------------------------------------------------------------

    async def aindex_documents(self, documents: List[Document], **kwargs) -> Any:
        """Async variant — awaits the async path."""
        return await self.index_documents(documents, **kwargs)

    async def index_documents(
        self,
        documents: List[Document],
        max_triplets_per_chunk: int = 15,
        title: Optional[str] = None,
        category: Optional[str] = None,
        kg_prompt_prefix: Optional[str] = None,
        num_passes: int = 1,
        similar_categories: Optional[List[str]] = None,
        guardrails: Optional[Dict] = None,
        small_chunk_size: int = 512,
        big_chunk_size: int = 1024,
        include_free_form: bool = False,
        agentic_chunk: bool = False,
    ) -> PropertyGraphIndex:
        """
        Main entry point for indexing documents into a PropertyGraphIndex.
        Uses small-to-big parsing and hybrid extraction.
        """
        if not documents:
            return self.index

        # 0b. Collect per-document summaries from metadata (set by indexer.py).
        # IMPORTANT: remove "summary" from doc.metadata BEFORE chunking.
        # SentenceSplitter measures metadata length against chunk_size; a 300-word
        # summary will exceed a 256-token small-chunk budget and raise ValueError.
        # We stash the summaries in a side-dict keyed by file path and re-attach
        # them to nodes only after chunking is complete.
        doc_summaries: Dict[str, str] = {}
        for doc in documents:
            # pop() removes "summary" from doc.metadata so SentenceSplitter never
            # measures it against chunk_size (a 300-word summary would exceed the
            # 256–512 token small-chunk budget and raise ValueError).
            summary = doc.metadata.pop("summary", "")
            if summary:
                src = doc.metadata.get("file_path") or doc.metadata.get("file_name", "")
                if src:
                    doc_summaries[src] = summary

        # 0. Context-aware entity alignment — use the document summary to rank
        #    existing graph entities by relevance, injecting only the top-K.
        first_summary = next(iter(doc_summaries.values()), None)
        existing_entities = self._fetch_existing_entities(
            category, similar_categories,
            document_summary=first_summary,
        )
        contextual_prefix = self._add_context_to_prefix(kg_prompt_prefix, existing_entities)

        # 1. Small-to-Big chunking (recursive + optional agentic)
        mode = "agentic + recursive" if agentic_chunk else "recursive section-aware"
        print(f"  -> Smart chunking ({mode}, small={small_chunk_size}, big={big_chunk_size})...")
        small_nodes, big_nodes = _small_to_big_parse(
            documents,
            small_chunk_size=small_chunk_size,
            big_chunk_size=big_chunk_size,
            agentic_chunk=agentic_chunk,
            llm=Settings.llm,
        )
        print(f"     {len(small_nodes)} small chunks, {len(big_nodes)} big (parent) chunks")

        # 1b. Apply contextual prefix + per-document summary to each small node.
        # Injecting the summary into every chunk ensures the extractor always has
        # full-document context, even when the active window is only 256 tokens.
        for node in small_nodes:
            node_src = node.metadata.get("file_path") or node.metadata.get("file_name", "")
            node_summary = doc_summaries.get(node_src, "")

            prefix_parts: list[str] = []
            if contextual_prefix:
                prefix_parts.append(contextual_prefix)
            if node_summary:
                summary_header = (
                    "[Document Summary – use to resolve ambiguous references]\n"
                    + node_summary
                )
                prefix_parts.append(summary_header)

            if prefix_parts:
                combined_prefix = "\n\n".join(prefix_parts)
                node.set_content(f"{combined_prefix}\n\n{node.get_content()}")

        # 2. Use LLM from global settings – resolve eagerly so extractors always
        #    receive a concrete instance, never None.
        llm = Settings.llm
        if llm is None:
            raise RuntimeError(
                "GraphIndexer.index_documents: Settings.llm is None. "
                "Configure Settings.llm before calling index_documents()."
            )

        # 3. Build extractor template
        extractors = _build_extractors_from_guardrails(
            guardrails, max_triplets_per_chunk, llm=llm, include_free_form=include_free_form
        )
        extractor_names = [type(e).__name__ for e in extractors]
        print(f"  -> Extractors: {', '.join(extractor_names)}")

        # 4. Initialize (or update) index with current extractors/transformations
        # Re-creating the index object ensures extractors and transformers are properly registered.
        self.index = PropertyGraphIndex(
            nodes=[],
            property_graph_store=self.storage_context.property_graph_store,
            kg_extractors=extractors,
            transformations=Settings.transformations,
            llm=llm,
            embed_model=Settings.embed_model,
            embed_kg_nodes=True,
            use_async=True,
            show_progress=False,
        )

        # Patch upsert_llama_nodes to guarantee ChunkNode always has a valid id_
        # Python's hash() can return negative numbers which Neo4j rejects as null
        import uuid as _uuid
        _original_upsert = self.index.property_graph_store.upsert_llama_nodes
        def _safe_upsert_llama_nodes(llama_nodes):
            for n in llama_nodes:
                if not n.id_:
                    n.id_ = str(_uuid.uuid4())
            return _original_upsert(llama_nodes)
        self.index.property_graph_store.upsert_llama_nodes = _safe_upsert_llama_nodes

        # Patch upsert_relations to drop any Relation whose source_id or target_id is
        # null — these cause Neo4j's MERGE to raise a SemanticError.
        _original_upsert_relations = self.index.property_graph_store.upsert_relations
        def _safe_upsert_relations(relations):
            valid = [
                r for r in relations
                if getattr(r, "source_id", None) and getattr(r, "target_id", None)
            ]
            skipped = len(relations) - len(valid)
            if skipped:
                print(f"     [DEBUG] upsert_relations: dropped {skipped} relation(s) with null source/target id", flush=True)
            if valid:
                return _original_upsert_relations(valid)
        self.index.property_graph_store.upsert_relations = _safe_upsert_relations

        # 5. Insertion passes — fully async to avoid deadlocking the event loop.
        # use_async=True on PropertyGraphIndex means insert_nodes internally schedules
        # coroutines; calling it synchronously inside an async context starves the loop.
        # ainsert_nodes properly yields control so Ollama HTTP calls can complete.

        def _sanitize_node(node):
            """Ensure a node has a valid id and clean relation/node metadata."""
            import uuid as _uuid
            from llama_index.core.graph_stores.types import KG_RELATIONS_KEY, KG_NODES_KEY, EntityNode, Relation

            if not node.node_id:
                node.id_ = str(_uuid.uuid4())

            # 1. Clean Relations
            raw_rels = node.metadata.get(KG_RELATIONS_KEY, [])
            clean_rels = []
            if isinstance(raw_rels, list):
                for r in raw_rels:
                    if isinstance(r, Relation):
                        src = getattr(r, "source_id", None)
                        tgt = getattr(r, "target_id", None)
                        if src and tgt and isinstance(src, str) and isinstance(tgt, str):
                            clean_rels.append(r)
            node.metadata[KG_RELATIONS_KEY] = clean_rels

            # 2. Clean Nodes (EntityNode)
            raw_kg_nodes = node.metadata.get(KG_NODES_KEY, [])
            clean_kg_nodes = []
            if isinstance(raw_kg_nodes, list):
                for n in raw_kg_nodes:
                    if isinstance(n, EntityNode):
                        name = getattr(n, "name", None)
                        if name and isinstance(name, str):
                            clean_kg_nodes.append(n)
            node.metadata[KG_NODES_KEY] = clean_kg_nodes

            return node

        async def _insert_pass(pass_num):
            total_failed = 0
            batch_size = max(1, (os.cpu_count() or 4) - 1)

            for i in range(0, len(small_nodes), batch_size):
                batch = small_nodes[i:i + batch_size]
                safe_nodes = [_sanitize_node(n) for n in batch]
                print(f"     [Pass {pass_num}] Processing batch of {len(safe_nodes)} chunks (offset {i})...", flush=True)
                try:
                    await self.index.ainsert_nodes(safe_nodes)
                except Exception as exc:
                    err_str = str(exc)
                    if "ConstraintValidationFailed" in err_str or "SemanticError" in err_str:
                        # Constraint errors often happen when parallel batches try to MERGE the same new entity.
                        # We fallback to sequentially inserting the nodes in this batch.
                        for node in safe_nodes:
                            try:
                                await self.index.ainsert_nodes([node])
                            except Exception as sub_exc:
                                if "ConstraintValidationFailed" not in str(sub_exc):
                                    total_failed += 1
                                    print(f"     [Pass {pass_num}] Sequential insert failed: {sub_exc}", flush=True)
                    else:
                        total_failed += len(safe_nodes)
                        print(f"     [Pass {pass_num}] Batch Insert failed: {exc}", flush=True)

            if total_failed:
                print(f"     ({total_failed} chunk(s) failed during pass {pass_num})")

        for i in range(num_passes):
            print(f"  -> PropertyGraph Extraction Pass {i + 1}/{num_passes} ({len(small_nodes)} nodes)...")
            try:
                await _insert_pass(i + 1)
            except Exception as e:
                err_str = str(e)
                if "ConstraintValidationFailed" in err_str:
                    print(f"     (Constraint error on pass {i + 1}, continuing)")
                else:
                    print(f"     Warning: Extraction pass error: {e}", flush=True)
                    import traceback
                    print(traceback.format_exc(), flush=True)

        # 5. Post-processing: collapse HAS_PROPERTY triplets into node props
        print("  -> Refining Business Object properties in Neo4j...")
        self._process_properties_in_graph()

        # 6. Promote entity_type → Neo4j label (enables per-type colors in Browser)
        #    and stamp category property so every node knows its document group.
        print("  -> Applying entity-type labels and category node...")
        self._apply_entity_labels(category)
        self._apply_category_node(category)

        # 7. Hybrid Graph Enhancements: Explicit Mentions and Semantic Edges
        self._link_chunks_to_entities()
        self._create_semantic_edges(small_nodes)

        return self.index

    # ------------------------------------------------------------------
    # Hybrid Graph Post-processing Helpers
    # ------------------------------------------------------------------

    def _link_chunks_to_entities(self):
        """
        Ensure explicit MENTIONS relationships between Chunks and Entities.
        LlamaIndex creates SOURCE relationships, but this makes it explicit for our Hybrid traversal.
        """
        if not isinstance(self.storage_context.property_graph_store, Neo4jPropertyGraphStore):
            return
            
        print("  -> Creating MENTIONS edges between Chunks and Entities...")
        # LlamaIndex usually connects __Node__ and __Entity__ with a SOURCE relationship
        query = """
        MATCH (c:__Node__)-[:SOURCE]-(e:__Entity__)
        MERGE (c)-[:MENTIONS]->(e)
        """
        try:
            self.storage_context.property_graph_store.structured_query(query)
            print("     ✓ MENTIONS edges created.")
        except Exception as e:
            print(f"     Warning: Failed to create MENTIONS edges: {e}")

    def _create_semantic_edges(self, new_nodes: Optional[List[TextNode]] = None, top_k: int = 3):
        """
        Calculates embeddings for the chunks and creates SIMILAR_TO edges
        to the most semantically similar existing chunks in the graph.
        
        If new_nodes is provided, only searches similarities for those specific nodes
        against the entire graph. If new_nodes is None, re-evaluates similarities
        for all nodes in the graph (Global Optimization).
        """
        if not isinstance(self.storage_context.property_graph_store, Neo4jPropertyGraphStore):
            return
            
        if not Settings.embed_model:
            print("  -> Skipping SIMILAR_TO edges (no embed_model configured).")
            return
            
        print(f"  -> Creating SIMILAR_TO semantic edges (top_k={top_k})...")
        
        # 1. Fetch all chunks and their stored embeddings (if any)
        query = """
        MATCH (c:__Node__) 
        WHERE c.text IS NOT NULL AND c.id IS NOT NULL 
        RETURN c.id AS id, c.text AS text, c.embedding AS embedding
        """
        results = self.storage_context.property_graph_store.structured_query(query)
        
        all_chunks = []
        if isinstance(results, list):
            for r in results:
                if isinstance(r, dict):
                    all_chunks.append({
                        "id": r.get("id"), 
                        "text": r.get("text"), 
                        "embedding": r.get("embedding")
                    })
                elif hasattr(r, "values"):
                    vals = list(r.values())
                    if len(vals) >= 2:
                        all_chunks.append({
                            "id": vals[0], 
                            "text": vals[1],
                            "embedding": vals[2] if len(vals) > 2 else None
                        })
                        
        if len(all_chunks) < 2:
            print("     Not enough chunks to create semantic edges.")
            return

        # 2. Compute missing embeddings
        import numpy as np
        
        # Track which chunks need updating in Neo4j
        chunks_to_update = []
        for chunk in all_chunks:
            if chunk["embedding"] is None:
                try:
                    emb = Settings.embed_model.get_text_embedding(chunk["text"])
                    chunk["embedding"] = emb
                    chunks_to_update.append({"id": chunk["id"], "embedding": emb})
                except Exception as e:
                    print(f"     Warning: Failed to embed chunk {chunk['id']}: {e}")
                    # Dynamically determine dimension from model if possible, fallback to 768
                    try:
                        dim = len(Settings.embed_model.get_query_embedding("test"))
                    except:
                        dim = 768
                    chunk["embedding"] = [0.0] * dim  # Dynamic dummy to prevent crash
        
        # Optional: Save embeddings back to Neo4j to save time on future runs
        if chunks_to_update:
            print(f"     Saving embeddings for {len(chunks_to_update)} chunk(s)...")
            # Batch update in Neo4j
            update_query = """
            UNWIND $updates AS update
            MATCH (c:__Node__) WHERE c.id = update.id
            SET c.embedding = update.embedding
            """
            try:
                # Chunk the updates to avoid query size limits
                batch_size = 100
                for i in range(0, len(chunks_to_update), batch_size):
                    batch = chunks_to_update[i:i+batch_size]
                    self.storage_context.property_graph_store.structured_query(
                        update_query, param_map={"updates": batch}
                    )
            except Exception as e:
                print(f"     Warning: Failed to save chunk embeddings: {e}")

        # 3. Filter all_chunks that have valid embeddings
        valid_chunks = [c for c in all_chunks if c["embedding"] is not None and len(c["embedding"]) > 1]
        if not valid_chunks:
            return

        # 4. Compute pairwise similarities:
        #    If new_nodes is provided, only check those. Otherwise, check all chunks.
        if new_nodes is not None:
            new_ids = {n.id_ for n in new_nodes}
            print(f"     Processing {len(new_ids)} new nodes...")
        else:
            new_ids = {c["id"] for c in valid_chunks}
            print(f"     Processing all {len(new_ids)} valid chunks...")
            
        emb_matrix = np.array([c["embedding"] for c in valid_chunks])
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        normed_emb_matrix = emb_matrix / norms
        
        edges_to_create = []
        
        for idx, chunk in enumerate(valid_chunks):
            if chunk["id"] not in new_ids:
                # Only create top-k edges *from* the newly indexed nodes
                continue
                
            chunk_emb = normed_emb_matrix[idx]
            # Compute similarity against all chunks
            similarities = normed_emb_matrix @ chunk_emb
            
            # Get top K + 1 (to exclude self-similarity)
            top_indices = np.argsort(similarities)[-(top_k+1):][::-1]
            
            added_edges = 0
            for target_idx in top_indices:
                if target_idx == idx:
                    continue  # Skip self
                
                sim_score = float(similarities[target_idx])
                if sim_score > 0.5: # Basic similarity threshold
                    edges_to_create.append({
                        "source_id": chunk["id"],
                        "target_id": valid_chunks[target_idx]["id"],
                        "score": sim_score
                    })
                    added_edges += 1
                
                if added_edges >= top_k:
                    break

        if edges_to_create:
            print(f"     Creating {len(edges_to_create)} SIMILAR_TO edges...")
            edge_query = """
            UNWIND $edges AS edge
            MATCH (source:__Node__) WHERE source.id = edge.source_id
            MATCH (target:__Node__) WHERE target.id = edge.target_id
            MERGE (source)-[r:SIMILAR_TO]->(target)
            SET r.score = edge.score
            """
            try:
                batch_size = 200
                for i in range(0, len(edges_to_create), batch_size):
                    batch = edges_to_create[i:i+batch_size]
                    self.storage_context.property_graph_store.structured_query(
                        edge_query, param_map={"edges": batch}
                    )
            except Exception as e:
                print(f"     Warning: Failed to create SIMILAR_TO edges: {e}")

    # ------------------------------------------------------------------
    # Post-processing helpers (unchanged from original)
    # ------------------------------------------------------------------

    def _process_properties_in_graph(self):
        """
        Finds all (n)-[HAS_PROPERTY]->(p) relationships and converts them
        to actual properties on node n.
        """
        if not isinstance(self.storage_context.property_graph_store, Neo4jPropertyGraphStore):
            return

        fetch_query = """
        MATCH (n)-[r:HAS_PROPERTY]->(p)
        RETURN elementId(n) as node_id, p.name as prop_raw, elementId(p) as prop_node_id, n.name as node_name
        """
        results = self.storage_context.property_graph_store.structured_query(fetch_query)

        if not results:
            return

        for record in results:
            node_id = record.get("node_id")
            prop_raw = record.get("prop_raw")
            prop_node_id = record.get("prop_node_id")
            node_name = record.get("node_name")

            if not prop_raw or ":" not in prop_raw:
                continue

            parts = prop_raw.split(":", 1)
            key = parts[0].strip().replace(" ", "_").lower()
            value = parts[1].strip()

            if key == "id":
                continue

            params = {"node_id": node_id, "val": value}
            if key in ["name", "title", "full_name"] and (not node_name or len(value) > len(node_name)):
                # Sanitize key name for safe interpolation as property key
                import re as _re
                key_safe = _re.sub(r'[^a-zA-Z0-9_]', '', key)
                set_query = f"MATCH (n) WHERE elementId(n) = $node_id SET n.name = $val, n.`{key_safe}` = $val"
            else:
                import re as _re
                key_safe = _re.sub(r'[^a-zA-Z0-9_]', '', key)
                set_query = f"MATCH (n) WHERE elementId(n) = $node_id SET n.`{key_safe}` = $val"

            try:
                self.storage_context.property_graph_store.structured_query(set_query, param_map=params)
                del_query = f"MATCH (p) WHERE elementId(p) = $prop_node_id DETACH DELETE p"
                self.storage_context.property_graph_store.structured_query(del_query, param_map={"prop_node_id": prop_node_id})
            except Exception as e:
                print(f"    Warning: Failed to refine property '{key}' for node {node_id}: {e}")

    # ------------------------------------------------------------------
    # Contextual alignment helpers
    # ------------------------------------------------------------------

    def _fetch_existing_entities(
        self,
        category: Optional[str],
        similar_categories: Optional[List[str]] = None,
        document_summary: Optional[str] = None,
        top_k: int = 20,
    ) -> List[str]:
        """
        Fetch semantically relevant existing entities from the graph.

        Instead of returning 150 random entities, this:
        1. Fetches all entity names + types from Neo4j.
        2. Embeds the current document's summary.
        3. Embeds all entity names.
        4. Returns only the top-K entities most similar to the document summary.

        This prevents polluting the extraction prompt with unrelated entities
        (e.g. finance entities when indexing a healthcare document).
        """
        cache_key = f"{category or 'global'}_{'_'.join(sorted(similar_categories or []))}"
        # Only use cache when there is no document_summary to rank against
        if document_summary is None and cache_key in self._entity_cache:
            return self._entity_cache[cache_key]

        if not isinstance(self.storage_context.property_graph_store, Neo4jPropertyGraphStore):
            return []

        query = """
        MATCH (n)
        WHERE (n.name IS NOT NULL OR n.id IS NOT NULL)
          AND NOT any(lbl IN labels(n) WHERE lbl IN ['__Community__', 'Category', 'CommunitySummary'])
        RETURN DISTINCT coalesce(n.name, n.id) as name,
               coalesce(n.title, n.name, n.id) as title,
               labels(n)[0] as type
        """
        results = self.storage_context.property_graph_store.structured_query(query)
        raw_entities: List[Dict[str, str]] = []
        if isinstance(results, list):
            for r in results:
                if isinstance(r, dict):
                    raw_entities.append({
                        "name": r.get("name", ""),
                        "title": r.get("title", r.get("name", "")),
                        "type": r.get("type", "Entity"),
                    })
                elif hasattr(r, "values"):
                    vals = list(r.values())
                    raw_entities.append({
                        "name": vals[0] if len(vals) > 0 else "",
                        "title": vals[1] if len(vals) > 1 else (vals[0] if vals else ""),
                        "type": vals[2] if len(vals) > 2 and vals[2] else "Entity",
                    })

        raw_entities = [e for e in raw_entities if e["name"]]
        if not raw_entities:
            return []

        # ── Semantic ranking (when summary is available) ──────────────
        if document_summary and Settings.embed_model is not None and len(raw_entities) > top_k:
            try:
                summary_emb = np.array(
                    Settings.embed_model.get_text_embedding(document_summary)
                )
                entity_labels = [
                    f"{e['title']} ({e['type']})" for e in raw_entities
                ]
                entity_embs = np.array([
                    Settings.embed_model.get_text_embedding(label)
                    for label in entity_labels
                ])
                # Cosine similarity: dot(a, B^T) / (|a| * |B|)
                norms = np.linalg.norm(entity_embs, axis=1)
                norms[norms == 0] = 1e-10
                summary_norm = np.linalg.norm(summary_emb)
                if summary_norm == 0:
                    summary_norm = 1e-10
                similarities = entity_embs @ summary_emb / (norms * summary_norm)
                top_indices = np.argsort(similarities)[-top_k:][::-1]

                ranked = [
                    f"{raw_entities[i]['title']} ({raw_entities[i]['type']})"
                    for i in top_indices
                ]
                print(
                    f"     [entity-injection] Ranked {len(raw_entities)} entities → top {len(ranked)} by relevance",
                    flush=True,
                )
                self._entity_cache[cache_key] = ranked
                return ranked
            except Exception as exc:
                print(
                    f"     [entity-injection] Semantic ranking failed ({exc}); falling back to unranked",
                    flush=True,
                )

        # ── Fallback: return all (capped) ─────────────────────────────
        fallback = [
            f"{e['title']} ({e['type']})" for e in raw_entities
        ][:top_k]
        self._entity_cache[cache_key] = fallback
        return fallback

    def _add_context_to_prefix(
        self, prefix: Optional[str], entities: List[str]
    ) -> Optional[str]:
        """Inject semantically-ranked existing entities into the prompt prefix."""
        if not prefix or not entities:
            return prefix

        context = (
            "IMPORTANT — Previously Extracted Entities (Reuse these EXACT names "
            "if referring to the same concept to avoid duplicates!):\n"
            + "\n".join(f"  • {e}" for e in entities)
            + "\n"
        )
        return prefix.replace(
            "Extract the triplets now",
            f"{context}\nExtract the triplets now",
        )

    # ------------------------------------------------------------------
    # Neo4j label & category node post-processing
    # ------------------------------------------------------------------

    def _apply_entity_labels(self, category: Optional[str]):
        """
        After insertion, promote each node's ``entity_type`` property into a
        real Neo4j label.  This is what makes Neo4j Browser color nodes by type.

        LlamaIndex stores everything under the generic ``__Entity__`` label;
        this pass adds the domain label (e.g. ``Company``, ``Person``) on top
        so the Browser's legend reflects actual entity types.

        Optionally also stamps the ``category`` property (document folder) onto
        every node that doesn't already have it.
        """
        if not isinstance(self.storage_context.property_graph_store, Neo4jPropertyGraphStore):
            return

        store = self.storage_context.property_graph_store

        # 1. Promote entity_type → Neo4j label for every node that has it.
        #    APOC is the cleanest way; fall back to a pure-Cypher loop when APOC
        #    is not available.
        apoc_query = """
        MATCH (n)
        WHERE n.entity_type IS NOT NULL AND n.entity_type <> ''
        WITH n, n.entity_type AS lbl
        CALL apoc.create.addLabels(n, [lbl]) YIELD node
        RETURN count(node) AS promoted
        """
        try:
            result = store.structured_query(apoc_query)
            count = 0
            if isinstance(result, list) and result:
                r = result[0]
                count = r.get("promoted", 0) if isinstance(r, dict) else (list(r.values())[0] if hasattr(r, "values") else 0)
            print(f"  -> [Labels] Promoted entity_type to Neo4j label for {count} node(s) via APOC.")
        except Exception:
            # APOC unavailable – collect distinct types and apply with CALL-IN-TRANSACTIONS
            print("  -> [Labels] APOC not available; using pure-Cypher label promotion...")
            type_query = "MATCH (n) WHERE n.entity_type IS NOT NULL RETURN DISTINCT n.entity_type AS t"
            type_results = store.structured_query(type_query)
            entity_types = []
            if isinstance(type_results, list):
                for r in type_results:
                    t = r.get("t") if isinstance(r, dict) else (list(r.values())[0] if hasattr(r, "values") else None)
                    if t:
                        entity_types.append(t)
            for et in entity_types:
                safe = et.replace("`", "")
                q = f"MATCH (n {{entity_type: '{safe}'}}) SET n:`{safe}`"
                try:
                    store.structured_query(q)
                except Exception as e:
                    print(f"     Warning: could not apply label '{safe}': {e}")
            print(f"  -> [Labels] Applied {len(entity_types)} entity-type label(s).")

        # 2. Stamp the category property on nodes that don't have it yet.
        if category:
            cat_safe = category.replace("'", "''")
            cat_q = f"""
            MATCH (n)
            WHERE n.entity_type IS NOT NULL AND (n.category IS NULL OR n.category = '')
            SET n.category = '{cat_safe}'
            """
            try:
                store.structured_query(cat_q)
            except Exception as e:
                print(f"     Warning: could not stamp category on nodes: {e}")

    def _apply_category_node(self, category: Optional[str]):
        """
        Create (or merge) a ``(:Category {{name: '<category>'}})`` anchor node and
        draw ``BELONGS_TO`` edges from every entity in the category to it.

        This gives you a top-level overview node in the Browser and lets you
        visually distinguish document groups with a dedicated color.
        """
        if not category:
            return
        if not isinstance(self.storage_context.property_graph_store, Neo4jPropertyGraphStore):
            return

        store = self.storage_context.property_graph_store
        cat_safe = category.replace("'", "''")

        # Ensure the Category node exists.
        merge_cat = f"MERGE (:Category {{name: '{cat_safe}'}})"
        try:
            store.structured_query(merge_cat)
        except Exception as e:
            print(f"     Warning: could not create Category node '{category}': {e}")
            return

        # Connect all entities that belong to this category.
        connect_q = f"""
        MATCH (n), (c:Category {{name: '{cat_safe}'}})
        WHERE n.category = '{cat_safe}'
          AND NOT (n)-[:BELONGS_TO]->(c)
          AND NOT n:Category
        MERGE (n)-[:BELONGS_TO]->(c)
        """
        try:
            store.structured_query(connect_q)
            print(f"  -> [Category] Linked entities to Category node '{category}'.")
        except Exception as e:
            print(f"     Warning: could not link entities to Category '{category}': {e}")

    def persist(self, persist_dir: str):
        """Neo4j handles its own persistence; no local files to save."""
        pass

    def load(self, persist_dir: str) -> bool:
        """Neo4j is live; index object is re-initialized during index_documents."""
        return True

    # ------------------------------------------------------------------
    # Graph cleanup (unchanged)
    # ------------------------------------------------------------------

    def clean_graph(self, similarity_threshold: float = 0.85, rel_threshold: float = None):
        """
        Refines the knowledge graph by merging nodes representing the same concept.
        """
        if not isinstance(self.storage_context.property_graph_store, Neo4jPropertyGraphStore):
            print("Cleanup only supported for Neo4jPropertyGraphStore.")
            return

        cleaner = GraphCleaner(
            self.storage_context.property_graph_store,
            embed_model=Settings.embed_model
        )
        cleaner.run_cleanup(similarity_threshold, rel_threshold=rel_threshold)

    def detect_and_summarize_communities(self) -> List[Document]:
        """
        Runs the full GraphRAG community detection and summarization pipeline.
        
        Steps:
        1. Build NetworkX graph from Neo4j.
        2. Detect communities using Louvain.
        3. Write community IDs back to Neo4j nodes.
        4. Generate LLM summaries for each community.
        5. Return summaries as LlamaIndex Documents for vector indexing.
        """
        if not isinstance(self.storage_context.property_graph_store, Neo4jPropertyGraphStore):
            print("Community detection only supported for Neo4jPropertyGraphStore.")
            return []

        store = self.storage_context.property_graph_store
        
        # 1. Build & Cluster
        G = build_networkx_graph(store)
        if G.number_of_nodes() == 0:
            print("     No nodes found in graph. Skipping community detection.")
            return []
            
        node_to_community = detect_communities(G)
        
        # 2. Persist community IDs
        write_communities_to_neo4j(store, node_to_community)
        
        # 3. Summarize
        summarize_communities(store, G, node_to_community)
        
        # 4. Export summaries as Documents
        summarizer = CommunitySummarizer(store)
        all_summaries = summarizer.get_all_summaries()
        
        summary_docs = []
        for s in all_summaries:
            cid = s.get("community_id")
            text = s.get("summary", "")
            entities = s.get("key_entities", "")
            count = s.get("entity_count", 0)
            
            doc_text = (
                f"Community {cid} (Theme Summary)\n"
                f"Size: {count} entities\n"
                f"Key Entities: {entities}\n\n"
                f"{text}"
            )
            
            metadata = {
                "source_type": "community_summary",
                "community_id": cid,
                "entity_count": count,
                "key_entities": entities,
            }
            
            summary_docs.append(Document(text=doc_text, metadata=metadata, id_=f"community_{cid}"))
            
        return summary_docs

    def refine_graph(self):
        """
        Perform a full suite of Knowledge Graph structural refinements.
        This updates labels, collapses properties, ensures MENTIONS links,
        and regenerates semantic chunk-to-chunk SIMILAR_TO edges.
        """
        if not isinstance(self.storage_context.property_graph_store, Neo4jPropertyGraphStore):
            print("Graph refinement only supported for Neo4jPropertyGraphStore.")
            return

        print("\n--- Refining Knowledge Graph Structure & Semantics ---")
        
        # 1. Promote HAS_PROPERTY to real Node properties
        print("  -> Refining Business Object properties...")
        self._process_properties_in_graph()
        
        # 2. Promote labels
        print("  -> Promoting entity_type → Neo4j labels...")
        self._apply_entity_labels(category=None) # Use category from props if any
        
        # 3. Explicit Mentions
        print("  -> Refreshing MENTIONS relationships...")
        self._link_chunks_to_entities()
        
        # 4. Semantic Edges (Global)
        print("  -> Re-computing semantic edges (SIMILAR_TO) for all chunks...")
        self._create_semantic_edges(new_nodes=None)
        
        print("✅ Graph refinement complete.")


# ===========================================================================
# GraphCleaner (unchanged)
# ===========================================================================