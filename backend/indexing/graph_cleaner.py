from llama_index.llms.ollama import Ollama
from datetime import datetime, timezone
import os
import re
import asyncio
import difflib
import nest_asyncio
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


class GraphCleaner:
    """Handles entity resolution/merging in Neo4j using semantic embeddings."""

    def __init__(self, graph_store: Neo4jPropertyGraphStore, embed_model: Any = None):
        self.graph_store = graph_store
        self.embed_model = embed_model or Settings.embed_model

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a, b = np.asarray(v1), np.asarray(v2)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def run_cleanup(self, similarity_threshold: float = 0.85, rel_threshold: float = None):
        """Main entry point for graph cleaning using semantic embeddings."""
        if rel_threshold is None:
            rel_threshold = similarity_threshold

        print(f"\n--- Starting Knowledge Graph Cleanup (Semantic Threshold: {similarity_threshold}) ---")

        nodes = self._fetch_all_nodes()
        if not nodes:
            print("No nodes found in graph.")
            return

        print(f"Total nodes to analyze: {len(nodes)}")

        # 0. Fast pass: merge nodes whose names are identical after case + whitespace normalisation.
        #    This catches the most common duplicate (same title, different casing) without
        #    needing embeddings.
        print("  -> Running case-insensitive exact-match dedup pass...")
        exact_clusters = self._cluster_exact_case_insensitive(nodes)
        if exact_clusters:
            print(f"     Found {len(exact_clusters)} exact-match cluster(s).")
            for cluster in exact_clusters:
                canonical = self._pick_canonical(cluster)
                duplicates = [n for n in cluster if n["id"] != canonical["id"]]
                dup_names = ", ".join([n["name"] for n in duplicates])
                print(f"     Merging (exact): [{dup_names}] -> [{canonical['name']}]")
                self._merge_nodes(canonical, duplicates)
            # Re-fetch after merges so the semantic pass works on the clean set.
            nodes = self._fetch_all_nodes()
            print(f"     Nodes remaining after exact-match pass: {len(nodes)}")

        if not nodes:
            self._merge_duplicate_relationships()
            print("--- Cleanup Complete ---\n")
            return

        # 1. Generate embeddings for all node names (batch where possible)
        print(f"  -> Generating embeddings for {len(nodes)} unique node names...")
        node_names = [n["name"] for n in nodes]
        try:
            # Simple iteration if the embed_model doesn't support batching well
            embeddings = []
            for i, name in enumerate(node_names):
                if i % 50 == 0 and i > 0:
                    print(f"     Embedded {i}/{len(nodes)}...")
                embeddings.append(self.embed_model.get_text_embedding(name))
            
            for i, node in enumerate(nodes):
                node["_embedding"] = embeddings[i]
        except Exception as e:
            print(f"    Warning: Embedding generation failed: {e}. Falling back to lexical matching.")
            for n in nodes:
                n["_embedding"] = None

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
        """Fetch all nodes with their IDs, names, and labels."""
        query = """
        MATCH (n)
        WHERE (n.name IS NOT NULL OR n.id IS NOT NULL)
          AND NOT any(lbl IN labels(n) WHERE lbl IN ['__Community__', 'Category', 'CommunitySummary'])
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

    def _cluster_exact_case_insensitive(self, nodes: List[Dict]) -> List[List[Dict]]:
        """
        Group nodes whose names are identical after lowering + stripping.
        Returns only groups with 2+ members (i.e. actual duplicates).
        """
        from collections import defaultdict
        buckets: Dict[str, List[Dict]] = defaultdict(list)
        for node in nodes:
            key = node["name"].strip().lower().replace(" ", "_")
            buckets[key].append(node)
        return [group for group in buckets.values() if len(group) > 1]

    def _cluster_similar_nodes(self, nodes: List[Dict], threshold: float) -> List[List[Dict]]:
        """
        Group nodes that are semantically similar using cosine similarity.

        Optimisations over the naive O(N²) approach:
        1. **Label Blocking** — only compare nodes that share at least one
           Neo4j label.  This drastically reduces comparisons when the graph
           contains many entity types (e.g. Person, Company, Technology).
        2. **Vectorised similarity** — when embeddings are available, stack
           them into a NumPy matrix and compute the full cosine similarity
           matrix in one shot (BLAS-accelerated).
        """
        from collections import defaultdict

        # ── Step 1: Group by label (blocking key) ──────────────────────
        label_buckets: Dict[str, List[int]] = defaultdict(list)
        for idx, node in enumerate(nodes):
            labels = node.get("labels", [])
            # Filter out generic LlamaIndex labels
            meaningful = [l for l in labels if l not in ("__Entity__", "__Node__")]
            if meaningful:
                for lbl in meaningful:
                    label_buckets[lbl].append(idx)
            else:
                # Nodes without meaningful labels go into a catch-all bucket
                label_buckets["__UNLABELED__"].append(idx)

        all_clusters: List[List[Dict]] = []
        globally_visited: set = set()

        for label, indices in label_buckets.items():
            bucket_nodes = [nodes[i] for i in indices]
            if len(bucket_nodes) < 2:
                continue

            # ── Step 2: Vectorised cosine similarity (per-block) ───────
            has_embeddings = all(n.get("_embedding") is not None for n in bucket_nodes)

            if has_embeddings:
                emb_matrix = np.array([n["_embedding"] for n in bucket_nodes])
                norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
                norms[norms == 0] = 1e-10
                normed = emb_matrix / norms
                sim_matrix = normed @ normed.T

                visited: set = set()
                for i in range(len(bucket_nodes)):
                    nid = bucket_nodes[i]["id"]
                    if nid in visited or nid in globally_visited:
                        continue
                    cluster = [bucket_nodes[i]]
                    visited.add(nid)
                    for j in range(i + 1, len(bucket_nodes)):
                        other_id = bucket_nodes[j]["id"]
                        if other_id in visited or other_id in globally_visited:
                            continue
                        if sim_matrix[i, j] >= threshold:
                            cluster.append(bucket_nodes[j])
                            visited.add(other_id)
                    if len(cluster) > 1:
                        all_clusters.append(cluster)
                        for n in cluster:
                            globally_visited.add(n["id"])
            else:
                # Fallback: lexical matching within the block
                visited = set()
                for i, node in enumerate(bucket_nodes):
                    if node["id"] in visited or node["id"] in globally_visited:
                        continue
                    cluster = [node]
                    visited.add(node["id"])
                    norm1 = node["name"].lower().strip()
                    for j in range(i + 1, len(bucket_nodes)):
                        other = bucket_nodes[j]
                        if other["id"] in visited or other["id"] in globally_visited:
                            continue
                        norm2 = other["name"].lower().strip()
                        sim = difflib.SequenceMatcher(None, norm1, norm2).ratio()
                        if sim >= threshold:
                            cluster.append(other)
                            visited.add(other["id"])
                    if len(cluster) > 1:
                        all_clusters.append(cluster)
                        for n in cluster:
                            globally_visited.add(n["id"])

        return all_clusters

    def _pick_canonical(self, cluster: List[Dict]) -> Dict:
        """
        Pick the 'best' node to keep.
        Strategy: Most relationships > Most properties > Shortest name.
        """
        best_node = cluster[0]
        max_score = -1

        for node in cluster:
            query = "MATCH (n)-[r]-() WHERE elementId(n) = $node_id RETURN count(r) as rel_count"
            result = self.graph_store.structured_query(query, param_map={"node_id": node["id"]})
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

            rel_types_query = "MATCH (d)-[r]-() WHERE elementId(d) = $dup_id RETURN DISTINCT type(r) as type"
            types_res = self.graph_store.structured_query(rel_types_query, param_map={"dup_id": dup_id})
            types = []
            if isinstance(types_res, list):
                for record in types_res:
                    if isinstance(record, dict):
                        types.append(record["type"])
                    elif hasattr(record, "values"):
                        types.append(record.values()[0])

            for t in types:
                t_safe = f"`{t}`"
                # Move outgoing relationships
                self.graph_store.structured_query(f"""
                MATCH (d)-[r:{t_safe}]->(target)
                WHERE elementId(d) = $dup_id AND elementId(target) <> $canonical_id
                MATCH (c) WHERE elementId(c) = $canonical_id
                MERGE (c)-[new_r:{t_safe}]->(target)
                SET new_r += properties(r)
                """, param_map={"dup_id": dup_id, "canonical_id": canonical_id})
                
                # Move incoming relationships
                self.graph_store.structured_query(f"""
                MATCH (source)-[r:{t_safe}]->(d)
                WHERE elementId(d) = $dup_id AND elementId(source) <> $canonical_id
                MATCH (c) WHERE elementId(c) = $canonical_id
                MERGE (source)-[new_r:{t_safe}]->(c)
                SET new_r += properties(r)
                """, param_map={"dup_id": dup_id, "canonical_id": canonical_id})

            # Merge properties using parameters
            d_props = dup.get("props", {})
            set_clauses = []
            params = {"canonical_id": canonical_id}
            for k, v in d_props.items():
                if k not in ["id", "name"]:
                    param_name = f"p_{k.replace(' ', '_')}"
                    set_clauses.append(f"c.`{k}` = ${param_name}")
                    params[param_name] = v

            if set_clauses:
                set_query = f"MATCH (c) WHERE elementId(c) = $canonical_id SET " + ", ".join(set_clauses)
                try:
                    self.graph_store.structured_query(set_query, param_map=params)
                except Exception as e:
                    print(f"    Warning: Failed to merge properties for {canonical_id}: {e}")

            self.graph_store.structured_query("MATCH (d) WHERE elementId(d) = $dup_id DETACH DELETE d", 
                                             param_map={"dup_id": dup_id})

    def _normalize_relationship_types(self, threshold: float):
        """Standardize similar relationship types globally using semantic embeddings."""
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

        # 1. Cleanse and normalize names for embedding/matching
        # We use a map to keep track of original names
        type_data = []
        for rt in rel_types:
            # Normalize for better embedding: 'WORKS_AT' -> 'works at'
            clean_name = rt.lower().replace("_", " ").strip()
            type_data.append({"original": rt, "clean": clean_name})

        clusters = []
        visited = set()

        # 2. Try semantic clustering if embed_model is available
        if self.embed_model:
            try:
                print(f"     Generating embeddings for {len(rel_types)} relationship types...")
                clean_names = [d["clean"] for d in type_data]
                # Relationship types are usually few, so we can embed them all
                embeddings = [self.embed_model.get_text_embedding(name) for name in clean_names]
                
                emb_matrix = np.array(embeddings)
                norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
                norms[norms == 0] = 1e-10
                normed = emb_matrix / norms
                sim_matrix = normed @ normed.T

                for i in range(len(type_data)):
                    orig_i = type_data[i]["original"]
                    if orig_i in visited:
                        continue
                    
                    cluster = [type_data[i]]
                    visited.add(orig_i)
                    
                    for j in range(i + 1, len(type_data)):
                        orig_j = type_data[j]["original"]
                        if orig_j in visited:
                            continue
                        
                        if sim_matrix[i, j] >= threshold:
                            cluster.append(type_data[j])
                            visited.add(orig_j)
                    
                    if len(cluster) > 1:
                        clusters.append([d["original"] for d in cluster])
            except Exception as e:
                print(f"     Warning: Semantic relationship normalization failed: {e}. Falling back to lexical.")
                visited = set() # Reset visited for lexical pass

        # 3. Lexical fallback/additional pass for remaining types
        if not clusters or len(visited) < len(rel_types):
            remaining_types = [d for d in type_data if d["original"] not in visited]
            if len(remaining_types) >= 2:
                for i, d_i in enumerate(remaining_types):
                    orig_i = d_i["original"]
                    if orig_i in visited:
                        continue
                    
                    cluster = [d_i]
                    visited.add(orig_i)
                    
                    for j in range(i + 1, len(remaining_types)):
                        d_j = remaining_types[j]
                        orig_j = d_j["original"]
                        if orig_j in visited:
                            continue
                        
                        sim = difflib.SequenceMatcher(None, d_i["clean"], d_j["clean"]).ratio()
                        if sim >= threshold:
                            cluster.append(d_j)
                            visited.add(orig_j)
                    
                    if len(cluster) > 1:
                        clusters.append([d["original"] for d in cluster])

        # 4. Perform the merges
        for cluster in clusters:
            # Pick canonical: shortest name usually preferred for types
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
        RETURN elementId(a) as source_id, elementId(b) as target_id, type
        """
        results = self.graph_store.structured_query(find_dups_query)
        if not results:
            return

        for record in results:
            source_id = record["source_id"]
            target_id = record["target_id"]
            rel_type = record["type"]

            print(f"    Merging duplicate '{rel_type}' rels between nodes.")
            merge_props_query = f"""
            MATCH (a)-[rs:`{rel_type}`]->(b)
            WHERE elementId(a) = $source_id AND elementId(b) = $target_id
            WITH rs
            ORDER BY elementId(rs) ASC
            WITH collect(rs) as rel_list
            WITH rel_list[0] as first_rel, rel_list[1..] as other_rels
            FOREACH (r IN other_rels | SET first_rel += properties(r) DELETE r)
            """
            try:
                self.graph_store.structured_query(merge_props_query, 
                                                 param_map={"source_id": source_id, "target_id": target_id})
            except Exception as e:
                print(f"    Warning: Failed to merge duplicate rels: {e}")