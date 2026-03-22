from llama_index.llms.ollama import Ollama
import os
import asyncio
import difflib
import nest_asyncio
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
from indexers.base import BaseIndexer

class RobustSchemaExtractor(SchemaLLMPathExtractor):
    """
    Bypasses SchemaLLMPathExtractor.__call__ entirely to avoid the
    asyncio.run() -> sniffio context loss chain on Python 3.14.
    Calls llm.structured_predict() (sync) directly instead.
    """

    def __call__(self, nodes, show_progress=False, **kwargs):
        from llama_index.core.graph_stores.types import KG_RELATIONS_KEY, Relation, EntityNode

        result_nodes = []
        for node in nodes:
            try:
                kg_schema = self.llm.structured_predict(
                    self.kg_schema_cls,
                    self.extract_prompt,
                    text=node.get_content(metadata_mode="llm"),
                )
                relations = []
                if kg_schema and hasattr(kg_schema, "triplets"):
                    for triplet in (kg_schema.triplets or []):
                        try:
                            subj_name = (triplet.subject.name or "").strip()
                            obj_name = (triplet.object.name or "").strip()
                            rel_type = (triplet.relation.type or "").strip()

                            if not subj_name or not obj_name or not rel_type:
                                print(f"     [DEBUG] Skipping empty triplet: {triplet}", flush=True)
                                continue

                            # Use deterministic, name-based IDs that match what
                            # EntityNode.id naturally produces: name.replace(" ", "_").lower()
                            # This ensures:
                            #   1. The same entity always merges to the same Neo4j node.
                            #   2. Relation source_id/target_id reference entities by name,
                            #      not random UUIDs, so Neo4j stores the entity name as the
                            #      node identity instead of a UUID.
                            subj_id = subj_name.replace(" ", "_").lower()
                            obj_id  = obj_name.replace(" ", "_").lower()

                            subj = EntityNode(
                                name=subj_name,
                                label=triplet.subject.type or "Entity",
                            )
                            obj = EntityNode(
                                name=obj_name,
                                label=triplet.object.type or "Entity",
                            )
                            rel = Relation(
                                source_id=subj_id,
                                target_id=obj_id,
                                label=rel_type,
                            )
                            relations.extend([subj, obj, rel])
                        except Exception as e:
                            print(f"     [DEBUG] Skipping triplet: {e}", flush=True)
                            continue
                node.metadata[KG_RELATIONS_KEY] = relations
                print(f"     [DEBUG] Extracted {len(relations)} relations from chunk", flush=True)
            except Exception as e:
                print(f"     [DEBUG] Extraction failed: {e}", flush=True)
                node.metadata[KG_RELATIONS_KEY] = []

            result_nodes.append(node)

        return result_nodes


# ---------------------------------------------------------------------------
# Small-to-Big Chunking Helper
# ---------------------------------------------------------------------------

def _small_to_big_parse(
    documents: List[Document],
    small_chunk_size: int = 256,
    small_chunk_overlap: int = 32,
    big_chunk_size: int = 1024,
    big_chunk_overlap: int = 128,
) -> Tuple[List[TextNode], List[TextNode]]:
    """
    Implements a 'small-to-big' parsing strategy:
    - **Small nodes** (256 tokens) for precise triplet extraction with minimal noise.
    - **Big nodes** (1024 tokens) retained as parent context for retrieval.

    Each small node stores a reference to its parent big node via
    ``node.metadata["parent_node_id"]``.

    Returns (small_nodes, big_nodes).
    """
    big_splitter = SentenceSplitter(
        chunk_size=big_chunk_size,
        chunk_overlap=big_chunk_overlap,
    )
    small_splitter = SentenceSplitter(
        chunk_size=small_chunk_size,
        chunk_overlap=small_chunk_overlap,
    )

    big_nodes: List[TextNode] = []
    small_nodes: List[TextNode] = []

    for doc in documents:
        # 1. Create big (parent) chunks
        parent_chunks = big_splitter.get_nodes_from_documents([doc])
        for parent in parent_chunks:
            big_nodes.append(parent)

            # 2. Create small (child) chunks from each parent
            child_chunks = small_splitter.get_nodes_from_documents(
                [Document(text=parent.text, metadata=parent.metadata)]
            )
            for child in child_chunks:
                if not child.node_id:
                    import uuid
                    child.id_ = str(uuid.uuid4())
                # Clear any relationship whose referenced node has a null id to prevent
                # ImplicitPathExtractor from creating Relations with null source/target ids.
                for rel_type in (
                    NodeRelationship.SOURCE,
                    NodeRelationship.NEXT,
                    NodeRelationship.PREVIOUS,
                ):
                    rel_info = child.relationships.get(rel_type)
                    if rel_info is not None and not rel_info.node_id:
                        child.relationships.pop(rel_type, None)
                child.metadata["parent_node_id"] = parent.node_id
                small_nodes.append(child)

    return small_nodes, big_nodes


# ---------------------------------------------------------------------------
# Schema helpers – build extractors from guardrail JSON
# ---------------------------------------------------------------------------

def _build_extractors_from_guardrails(
    guardrails: Optional[Dict],
    max_triplets_per_chunk: int = 15,
    llm: Optional[Any] = None,
    include_free_form: bool = False,
) -> list:
    """
    Build a list of KG extractors for PropertyGraphIndex based on guardrails.

    Strategy:
    1. **SchemaLLMPathExtractor** – constrained to the entity/relationship
       types declared in the guardrails (schema-guided).
    2. **SimpleLLMPathExtractor** – (Optional) fallback free-form extraction.
    3. **ImplicitPathExtractor** – captures document-structural relations
       (NEXT, PREVIOUS, SOURCE) for free (no LLM calls).
    """
    if llm is None:
        llm = Settings.llm

    if llm is None:
        raise ValueError(
            "_build_extractors_from_guardrails: llm is None. "
            "Ensure Settings.llm is set before calling index_documents(), "
            "or pass the llm argument explicitly."
        )

    extractors = []

    # 1. Schema-guided extractor (primary)
    if guardrails:
        # Collect entity types from business_objects or entity_types
        business_objects = guardrails.get("business_objects", [])
        entity_types_from_objects = [obj["name"] for obj in business_objects if "name" in obj]
        entity_types = entity_types_from_objects or guardrails.get("entity_types", ["Entity"])

        relationship_types_raw = guardrails.get("relationship_types", ["RELATED_TO"])
        relationship_types = []
        for r in relationship_types_raw:
            if isinstance(r, str):
                relationship_types.append(r)
            elif isinstance(r, dict) and "name" in r:
                relationship_types.append(r["name"])

        extractors.append(
            RobustSchemaExtractor(
                llm=llm,
                possible_entities=entity_types,
                possible_relations=relationship_types,
                strict=False,
                max_triplets_per_chunk=max_triplets_per_chunk,
            )
        )

    # 2. Free-form fallback extractor (Optional/Conditional)
    # If we have guardrails AND include_free_form=False, we skip this to save LLM calls.
    if not guardrails or include_free_form:
        extractors.append(
            SimpleLLMPathExtractor(
                llm=llm,
                max_paths_per_chunk=max(5, max_triplets_per_chunk // 3),
            )
        )

    # 3. Implicit structural extractor (no LLM cost)
    extractors.append(ImplicitPathExtractor())

    return extractors


# ===========================================================================
# GraphIndexer
# ===========================================================================

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

    def index_documents(
        self,
        documents: List[Document],
        max_triplets_per_chunk: int = 15,
        title: Optional[str] = None,
        category: Optional[str] = None,
        kg_prompt_prefix: Optional[str] = None,
        num_passes: int = 1,
        similar_categories: Optional[List[str]] = None,
        guardrails: Optional[Dict] = None,
        small_chunk_size: int = 256,
        big_chunk_size: int = 1024,
        include_free_form: bool = False,
    ) -> PropertyGraphIndex:
        """
        Main entry point for indexing documents into a PropertyGraphIndex.
        Uses small-to-big parsing and hybrid extraction.
        """
        if not documents:
            return self.index

        # 0. Contextual alignment – existing entity names
        existing_entities = self._fetch_existing_entities(category, similar_categories)
        contextual_prefix = self._add_context_to_prefix(kg_prompt_prefix, existing_entities)

        # 1. Small-to-Big chunking
        print(f"  -> Small-to-big parsing (small={small_chunk_size}, big={big_chunk_size})...")
        small_nodes, big_nodes = _small_to_big_parse(
            documents, # Use documents directly, assuming they are enriched by caller
            small_chunk_size=small_chunk_size,
            big_chunk_size=big_chunk_size,
        )
        print(f"     {len(small_nodes)} small chunks, {len(big_nodes)} big (parent) chunks")

        # 1b. Apply contextual prefix to each small node *after* chunking
        if contextual_prefix:
            for node in small_nodes:
                node.set_content(f"{contextual_prefix}\n\n{node.get_content()}")

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
            use_async=False,
            show_progress=True,
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

        # 5. Insertion passes
        # SchemaLLMPathExtractor relies on anyio/sniffio to detect the async backend.
        # sniffio checks a context variable that is ONLY set when running inside an
        # anyio-managed task. asyncio.run / loop.run_until_complete don't set it,
        # causing "unknown async library". Fix: use anyio.run(..., backend="asyncio")
        # which correctly sets the sniffio context. nest_asyncio.apply() lets
        # anyio.run work even when an event loop is already running (e.g. Jupyter).
        import anyio
        import asyncio
        nest_asyncio.apply()
        index_ref = self.index

        async def _insert_pass():
            failed = 0
            for node in small_nodes:
                print(f"     [DEBUG] node.id_={repr(node.id_)} node.node_id={repr(node.node_id)}", flush=True)
                try:
                    # Ensure node has a valid id before inserting
                    if not node.node_id:
                        import uuid
                        node.id_ = str(uuid.uuid4())

                    # Filter out any relations with null source/target before insertion
                    # so ImplicitPathExtractor results don't slip through as null ids.
                    from llama_index.core.graph_stores.types import KG_RELATIONS_KEY, KG_NODES_KEY
                    raw_rels = node.metadata.get(KG_RELATIONS_KEY, [])
                    clean_rels = []
                    for r in raw_rels:
                        src = getattr(r, "source_id", None)
                        tgt = getattr(r, "target_id", None)
                        nid = getattr(r, "id", None)
                        # Relation objects need both src+tgt; EntityNode objects need id
                        if src is not None and tgt is not None:
                            clean_rels.append(r)
                        elif src is None and tgt is None and nid is not None:
                            clean_rels.append(r)
                    if len(clean_rels) < len(raw_rels):
                        print(f"     [DEBUG] Dropped {len(raw_rels) - len(clean_rels)} null-id relation(s) from KG_RELATIONS_KEY", flush=True)
                    node.metadata[KG_RELATIONS_KEY] = clean_rels

                    # Print every relation object to find the null id
                    for rel in node.metadata.get(KG_RELATIONS_KEY, []):
                        print(f"     [DEBUG] REL source_id={repr(getattr(rel, 'source_id', 'N/A'))} target_id={repr(getattr(rel, 'target_id', 'N/A'))} label={repr(getattr(rel, 'label', 'N/A'))} id={repr(getattr(rel, 'id', 'N/A'))}", flush=True)
                    for n in node.metadata.get(KG_NODES_KEY, []):
                        print(f"     [DEBUG] NODE id={repr(getattr(n, 'id', 'N/A'))} name={repr(getattr(n, 'name', 'N/A'))}", flush=True)

                    index_ref.insert_nodes([node])
                except Exception as exc:
                    if "ConstraintValidationFailed" in str(exc):
                        pass
                    else:
                        failed += 1
                        print(f"     [DEBUG] Insert failed: {exc}", flush=True)
                        import traceback
                        print(traceback.format_exc(), flush=True)
            if failed:
                print(f"     ({failed} chunk(s) failed during insert)")
                

        for i in range(num_passes):
            print(f"  -> PropertyGraph Extraction Pass {i + 1}/{num_passes}...")
            try:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(_insert_pass())
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

        return self.index

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

            if key in ["name", "title", "full_name"] and (not node_name or len(value) > len(node_name)):
                set_query = f"""
                MATCH (n) WHERE elementId(n) = '{node_id}' 
                SET n.name = '{value.replace("'", "''")}', n.{key} = '{value.replace("'", "''")}'
                """
            else:
                escaped_val = value.replace("'", "''")
                set_query = f"MATCH (n) WHERE elementId(n) = '{node_id}' SET n.{key} = '{escaped_val}'"

            try:
                self.storage_context.property_graph_store.structured_query(set_query)
                del_query = f"MATCH (p) WHERE elementId(p) = '{prop_node_id}' DETACH DELETE p"
                self.storage_context.property_graph_store.structured_query(del_query)
            except Exception as e:
                print(f"    Warning: Failed to refine property '{key}' for node {node_id}: {e}")

    # ------------------------------------------------------------------
    # Contextual alignment helpers
    # ------------------------------------------------------------------

    def _fetch_existing_entities(
        self,
        category: Optional[str],
        similar_categories: Optional[List[str]] = None,
    ) -> List[str]:
        """Fetch names of existing entities in this category or similar categories/graph."""
        cache_key = f"{category or 'global'}_{'_'.join(sorted(similar_categories or []))}"
        if cache_key in self._entity_cache:
            return self._entity_cache[cache_key]

        if not isinstance(self.storage_context.property_graph_store, Neo4jPropertyGraphStore):
            return []

        query = """
        MATCH (n) 
        WHERE (n.name IS NOT NULL OR n.id IS NOT NULL) 
        RETURN DISTINCT coalesce(n.name, n.id) as name, labels(n)[0] as type 
        LIMIT 150
        """
        results = self.storage_context.property_graph_store.structured_query(query)
        entities = []
        if isinstance(results, list):
            for r in results:
                if isinstance(r, dict):
                    entities.append(f"{r['name']} ({r.get('type', 'Entity')})")
                elif hasattr(r, "values"):
                    vals = list(r.values())
                    type_str = vals[1] if len(vals) > 1 and vals[1] else "Entity"
                    entities.append(f"{vals[0]} ({type_str})")

        processed_entities = [e for e in entities if e]
        self._entity_cache[cache_key] = processed_entities
        return processed_entities

    def _add_context_to_prefix(self, prefix: Optional[str], entities: List[str]) -> Optional[str]:
        """Inject existing entities into the prompt prefix for alignment."""
        if not prefix or not entities:
            return prefix

        context = (
            "Recently Extracted Entities (Use exactly these names if referring to the same concept!): "
            + ", ".join(entities[:20])
            + "\n"
        )
        return prefix.replace("Extract the triplets now", f"{context}\nExtract the triplets now")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist(self, persist_dir: str):
        """Neo4j handles its own persistence; no local files to save."""
        pass

    def load(self, persist_dir: str) -> bool:
        """Neo4j is live; index object is re-initialized during index_documents."""
        return True

    # ------------------------------------------------------------------
    # Graph cleanup (unchanged)
    # ------------------------------------------------------------------

    def clean_graph(self, similarity_threshold: float = 0.9, rel_threshold: float = None):
        """
        Refines the knowledge graph by merging nodes representing the same concept.
        """
        if not isinstance(self.storage_context.property_graph_store, Neo4jPropertyGraphStore):
            print("Cleanup only supported for Neo4jPropertyGraphStore.")
            return

        cleaner = GraphCleaner(self.storage_context.property_graph_store)
        cleaner.run_cleanup(similarity_threshold, rel_threshold=rel_threshold)


# ===========================================================================
# GraphCleaner (unchanged)
# ===========================================================================

class GraphCleaner:
    """Handles entity resolution/merging in Neo4j."""

    def __init__(self, graph_store: Neo4jPropertyGraphStore):
        self.graph_store = graph_store

    def run_cleanup(self, similarity_threshold: float = 0.9, rel_threshold: float = None):
        """Main entry point for graph cleaning."""
        if rel_threshold is None:
            rel_threshold = similarity_threshold

        print(f"\n--- Starting Knowledge Graph Cleanup (Node Threshold: {similarity_threshold}, Rel Threshold: {rel_threshold}) ---")

        nodes = self._fetch_all_nodes()
        if not nodes:
            print("No nodes found in graph.")
            return

        print(f"Total nodes to analyze: {len(nodes)}")

        self._normalize_relationship_types(rel_threshold)

        clusters = self._cluster_similar_nodes(nodes, similarity_threshold)
        if not clusters:
            print("No duplicates detected.")
            return

        print(f"Found {len(clusters)} cluster(s) of potential duplicates.")

        for cluster in clusters:
            canonical = self._pick_canonical(cluster)
            duplicates = [n for n in cluster if n["id"] != canonical["id"]]

            duplicate_names = ", ".join([n["name"] for n in duplicates])
            print(f"  -> Merging: [{duplicate_names}] -> [{canonical['name']}]")

            self._merge_nodes(canonical, duplicates)

        self._merge_duplicate_relationships()

        print("--- Cleanup Complete ---\n")

    def _fetch_all_nodes(self) -> List[Dict]:
        """Fetch all nodes with their IDs, names, and labels.

        Excludes llama-index internal infrastructure nodes (__Entity__, __Node__,
        __Chunk__, __Community__) which have no meaningful name and are not KG
        entities — including them causes the cleanup to see an empty result set.
        """
        query = """
        MATCH (n)
        WHERE (n.name IS NOT NULL OR n.id IS NOT NULL)
          AND NOT any(lbl IN labels(n) WHERE lbl IN ['__Community__'])
        RETURN elementId(n) AS id, coalesce(n.name, n.id) AS name, labels(n) AS labels, properties(n) as props
        """
        result = self.graph_store.structured_query(query)
        nodes = []
        if isinstance(result, list):
            for record in result:
                if isinstance(record, dict):
                    nodes.append(record)
                elif hasattr(record, "values"):
                    values = record.values()
                    nodes.append({"id": values[0], "name": values[1], "labels": values[2], "props": values[3]})
        return [n for n in nodes if n.get("name")]

    def _cluster_similar_nodes(self, nodes: List[Dict], threshold: float) -> List[List[Dict]]:
        """Group nodes that are string-similar after normalization."""
        clusters = []
        visited = set()

        for n in nodes:
            n["_norm"] = n["name"].lower().strip()

        for i, node in enumerate(nodes):
            if node["id"] in visited:
                continue

            current_cluster = [node]
            visited.add(node["id"])

            for j in range(i + 1, len(nodes)):
                other = nodes[j]
                if other["id"] in visited:
                    continue

                sim = difflib.SequenceMatcher(None, node["_norm"], other["_norm"]).ratio()
                if sim >= threshold:
                    current_cluster.append(other)
                    visited.add(other["id"])

            if len(current_cluster) > 1:
                clusters.append(current_cluster)

        return clusters

    def _pick_canonical(self, cluster: List[Dict]) -> Dict:
        """
        Pick the 'best' node to keep.
        Strategy: Most relationships > Most properties > Shortest name.
        """
        best_node = cluster[0]
        max_score = -1

        for node in cluster:
            query = f"MATCH (n)-[r]-() WHERE elementId(n) = '{node['id']}' RETURN count(r) as rel_count"
            result = self.graph_store.structured_query(query)
            rel_count = 0
            try:
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict):
                        rel_count = result[0].get("rel_count", 0)
                    elif hasattr(result[0], "values"):
                        rel_count = result[0].values()[0]
            except Exception:
                pass

            prop_count = len(node.get("props", {}))
            score = (rel_count * 10) + prop_count

            if score > max_score:
                max_score = score
                best_node = node
            elif score == max_score:
                if len(node["name"]) < len(best_node["name"]):
                    best_node = node

        return best_node

    def _merge_nodes(self, canonical: Dict, duplicates: List[Dict]):
        """
        Merge duplicate nodes into the canonical node.
        """
        canonical_id = canonical["id"]

        for dup in duplicates:
            dup_id = dup["id"]

            rel_types_query = f"MATCH (d)-[r]-() WHERE elementId(d) = '{dup_id}' RETURN DISTINCT type(r) as type"
            types_res = self.graph_store.structured_query(rel_types_query)
            types = []
            if isinstance(types_res, list):
                for record in types_res:
                    if isinstance(record, dict):
                        types.append(record["type"])
                    elif hasattr(record, "values"):
                        types.append(record.values()[0])

            for t in types:
                t_safe = f"`{t}`"
                self.graph_store.structured_query(f"""
                MATCH (d)-[r:{t_safe}]->(target)
                WHERE elementId(d) = '{dup_id}' AND elementId(target) <> '{canonical_id}'
                MATCH (c) WHERE elementId(c) = '{canonical_id}'
                MERGE (c)-[new_r:{t_safe}]->(target)
                SET new_r += properties(r)
                """)
                self.graph_store.structured_query(f"""
                MATCH (source)-[r:{t_safe}]->(d)
                WHERE elementId(d) = '{dup_id}' AND elementId(source) <> '{canonical_id}'
                MATCH (c) WHERE elementId(c) = '{canonical_id}'
                MERGE (source)-[new_r:{t_safe}]->(c)
                SET new_r += properties(r)
                """)

            d_props = dup.get("props", {})
            set_clauses = []
            for k, v in d_props.items():
                if k not in ["id", "name"]:
                    val_str = str(v).replace("'", "''")
                    set_clauses.append(f"c.{k} = '{val_str}'")

            if set_clauses:
                set_query = f"MATCH (c) WHERE elementId(c) = '{canonical_id}' SET " + ", ".join(set_clauses)
                try:
                    self.graph_store.structured_query(set_query)
                except Exception as e:
                    print(f"    Warning: Failed to merge properties for {canonical_id}: {e}")

            self.graph_store.structured_query(f"MATCH (d) WHERE elementId(d) = '{dup_id}' DETACH DELETE d")

    def _normalize_relationship_types(self, threshold: float):
        """Standardize similar sounding relationship types globally."""
        print("  -> Normalizing relationship types...")
        query = "CALL db.relationshipTypes()"
        result = self.graph_store.structured_query(query)
        rel_types = []
        if isinstance(result, list):
            for record in result:
                if isinstance(record, dict):
                    rel_types.append(record["relationshipType"])
                elif hasattr(record, "values"):
                    rel_types.append(record.values()[0])

        if len(rel_types) < 2:
            return

        clusters = []
        visited = set()

        normalized_map = {rt: rt.upper().replace("_", " ").strip() for rt in rel_types}

        sorted_types = sorted(rel_types, key=len)

        for i, rt in enumerate(sorted_types):
            if rt in visited:
                continue

            cluster = [rt]
            visited.add(rt)

            norm_rt = normalized_map[rt]

            for j in range(i + 1, len(sorted_types)):
                other = sorted_types[j]
                if other in visited:
                    continue

                norm_other = normalized_map[other]

                sim = difflib.SequenceMatcher(None, norm_rt, norm_other).ratio()
                if sim >= threshold:
                    cluster.append(other)
                    visited.add(other)

            if len(cluster) > 1:
                clusters.append(cluster)

        for cluster in clusters:
            canonical = sorted(cluster, key=len)[0]
            duplicates = [rt for rt in cluster if rt != canonical]

            for dup in duplicates:
                print(f"    Merging relationship types: [{dup}] -> [{canonical}]")
                merge_query = f"""
                MATCH (a)-[r:`{dup}`]->(b)
                WITH a, b, properties(r) as props, r
                MERGE (a)-[new_r:`{canonical}`]->(b)
                SET new_r += props
                DELETE r
                """
                try:
                    self.graph_store.structured_query(merge_query)
                except Exception as e:
                    print(f"    Warning: Failed to merge relationship type {dup}: {e}")

    def _merge_duplicate_relationships(self):
        """Find nodes with multiple relationships of the same type between them and merge them."""
        print("  -> Merging duplicate relationships between same node pairs...")
        find_dups_query = """
        MATCH (a)-[r]->(b)
        WITH a, b, type(r) as type, count(r) as count, collect(r) as rels
        WHERE count > 1
        RETURN elementId(a) as source_id, elementId(b) as target_id, type, rels[0] as canonical_rel, rels[1..] as duplicates
        """
        results = self.graph_store.structured_query(find_dups_query)
        if not results:
            return

        for record in results:
            canonical_rel = record.get("canonical_rel")
            duplicates = record.get("duplicates", [])

            source_id = record["source_id"]
            target_id = record["target_id"]
            rel_type = record["type"]

            print(f"    Merging {len(duplicates)} duplicate '{rel_type}' rels between nodes.")

            merge_props_query = f"""
            MATCH (a)-[rs:`{rel_type}`]->(b)
            WHERE elementId(a) = '{source_id}' AND elementId(b) = '{target_id}'
            WITH rs
            ORDER BY elementId(rs) ASC
            WITH collect(rs) as rel_list
            WITH rel_list[0] as first_rel, rel_list[1..] as other_rels
            FOREACH (r IN other_rels | SET first_rel += properties(r) DELETE r)
            """
            try:
                self.graph_store.structured_query(merge_props_query)
            except Exception as e:
                print(f"    Warning: Failed to merge duplicate rels: {e}")