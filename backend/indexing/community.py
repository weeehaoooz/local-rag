"""
indexers/community.py
---------------------
CommunitySummarizer: generates LLM summaries for detected graph communities
and stores them as (:CommunitySummary) nodes in Neo4j.

Each CommunitySummary node contains:
  - community_id (int)
  - summary (str)       — LLM-generated description of the community
  - entity_count (int)  — number of entities in the community
  - key_entities (str)  — comma-separated list of top entity names

These nodes are linked to their constituent entities via:
  (entity)-[:MEMBER_OF]->(community_summary)
"""

from __future__ import annotations

import os
from typing import List, Optional
from collections import defaultdict
import networkx as nx

from llama_index.core import Settings
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore


class CommunitySummarizer:
    """
    Generate and store LLM summaries for Knowledge Graph communities.

    Usage::

        summarizer = CommunitySummarizer(graph_store)
        summarizer.summarize_and_store(
            community_id=0,
            entity_names=["Apple Inc.", "Steve Jobs", "Cupertino"],
            relationships=["Apple Inc. --[FOUNDED_BY]--> Steve Jobs", ...],
        )
    """

    def __init__(
        self,
        graph_store: Neo4jPropertyGraphStore,
        llm=None,
    ):
        self.graph_store = graph_store
        self.llm = llm or Settings.llm

    def summarize_and_store(
        self,
        community_id: int,
        entity_names: List[str],
        relationships: List[str],
        max_entities_in_prompt: int = 50,
        max_rels_in_prompt: int = 80,
    ) -> str:
        """
        Generate a summary for a community and persist it in Neo4j.

        Parameters
        ----------
        community_id : int
            The numeric community identifier.
        entity_names : list[str]
            Names of all entities in this community.
        relationships : list[str]
            Human-readable relationship descriptions (e.g. "A --[REL]--> B").
        max_entities_in_prompt : int
            Cap on entities to include in the LLM prompt.
        max_rels_in_prompt : int
            Cap on relationships to include in the LLM prompt.

        Returns
        -------
        str
            The generated summary.
        """
        # Deduplicate
        unique_entities = list(dict.fromkeys(entity_names))[:max_entities_in_prompt]
        unique_rels = list(dict.fromkeys(relationships))[:max_rels_in_prompt]

        entities_str = ", ".join(unique_entities)
        rels_str = "\n".join(f"  - {r}" for r in unique_rels)

        prompt = (
            "You are a Knowledge Graph analyst. Below is a cluster (community) of "
            "related entities and their relationships extracted from a knowledge graph.\n\n"
            f"ENTITIES ({len(unique_entities)}): {entities_str}\n\n"
            f"RELATIONSHIPS:\n{rels_str}\n\n"
            "Write a concise but comprehensive summary (150-250 words) that:\n"
            "1. Identifies the main theme or domain this community represents.\n"
            "2. Lists the most important entities and their roles.\n"
            "3. Describes the key relationships and how entities are connected.\n"
            "4. Notes any interesting patterns or hierarchies.\n\n"
            "Do NOT include any preamble. Start directly with the summary."
        )

        try:
            response = self.llm.complete(prompt)
            summary = response.text.strip()
        except Exception as e:
            print(f"     Warning: LLM summarization failed for community {community_id}: {e}")
            summary = f"Community of {len(unique_entities)} entities: {entities_str[:200]}"

        # Key entities = first 10
        key_entities = ", ".join(unique_entities[:10])

        # Store in Neo4j
        self._upsert_community_node(
            community_id=community_id,
            summary=summary,
            entity_count=len(unique_entities),
            key_entities=key_entities,
        )

        # Link entities to the community summary node
        self._link_entities_to_community(community_id)

        print(
            f"     Community {community_id}: {len(unique_entities)} entities, "
            f"{len(unique_rels)} rels — summarized ({len(summary)} chars)"
        )
        return summary

    def _upsert_community_node(
        self,
        community_id: int,
        summary: str,
        entity_count: int,
        key_entities: str,
    ):
        """Create or update a (:CommunitySummary) node in Neo4j."""
        query = """
        MERGE (cs:CommunitySummary {community_id: $cid})
        SET cs.summary = $summary,
            cs.entity_count = $entity_count,
            cs.key_entities = $key_entities,
            cs.name = 'Community ' + toString($cid)
        """
        try:
            self.graph_store.structured_query(
                query,
                param_map={
                    "cid": community_id,
                    "summary": summary,
                    "entity_count": entity_count,
                    "key_entities": key_entities,
                },
            )
        except Exception as e:
            print(f"     Warning: Failed to upsert CommunitySummary {community_id}: {e}")

    def _link_entities_to_community(self, community_id: int):
        """
        Create (entity)-[:MEMBER_OF]->(CommunitySummary) edges for all entities
        in this community.
        """
        query = """
        MATCH (n {community_id: $cid}), (cs:CommunitySummary {community_id: $cid})
        WHERE NOT n:CommunitySummary
          AND NOT (n)-[:MEMBER_OF]->(cs)
        MERGE (n)-[:MEMBER_OF]->(cs)
        """
        try:
            self.graph_store.structured_query(
                query, param_map={"cid": community_id}
            )
        except Exception as e:
            print(f"     Warning: Failed to link entities to community {community_id}: {e}")

    def get_all_summaries(self) -> list[dict]:
        """Retrieve all community summaries from Neo4j."""
        query = """
        MATCH (cs:CommunitySummary)
        RETURN cs.community_id AS community_id,
               cs.summary AS summary,
               cs.entity_count AS entity_count,
               cs.key_entities AS key_entities
        ORDER BY cs.entity_count DESC
        """
        results = self.graph_store.structured_query(query)
        summaries = []
        if isinstance(results, list):
            for r in results:
                if isinstance(r, dict):
                    summaries.append(r)
                elif hasattr(r, "values"):
                    vals = list(r.values())
                    summaries.append({
                        "community_id": vals[0],
                        "summary": vals[1],
                        "entity_count": vals[2],
                        "key_entities": vals[3],
                    })
        return summaries

    def get_relevant_summaries(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Retrieve the community summaries most relevant to a user query
        using embedding-based semantic search.

        Falls back to returning the largest communities if embedding fails.
        """
        all_summaries = self.get_all_summaries()
        if not all_summaries:
            return []

        embed_model = Settings.embed_model
        if embed_model is None:
            # Fallback: return largest communities
            return all_summaries[:top_k]

        try:
            import numpy as np

            query_emb = np.array(embed_model.get_text_embedding(query))
            summary_texts = [s.get("summary", "") for s in all_summaries]
            summary_embs = np.array([
                embed_model.get_text_embedding(text) for text in summary_texts
            ])

            norms = np.linalg.norm(summary_embs, axis=1)
            norms[norms == 0] = 1e-10
            query_norm = np.linalg.norm(query_emb)
            if query_norm == 0:
                query_norm = 1e-10

            similarities = summary_embs @ query_emb / (norms * query_norm)
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            return [all_summaries[i] for i in top_indices]
        except Exception as e:
            print(f"     Warning: Semantic community search failed: {e}")
            return all_summaries[:top_k]

# --- Clustering Utilities (Moved from scripts/detect_communities.py) ---

def build_networkx_graph(graph_store: Neo4jPropertyGraphStore) -> nx.Graph:
    """
    Pull all nodes and relationships from Neo4j and build a NetworkX graph.
    Each NX node carries the Neo4j elementId, name, labels, and properties.
    """
    print("  -> Fetching nodes from Neo4j...")
    node_query = """
    MATCH (n)
    WHERE n.name IS NOT NULL
      AND NOT any(lbl IN labels(n) WHERE lbl IN ['__Community__', 'Category', 'CommunitySummary'])
    RETURN elementId(n) AS id, n.name AS name, labels(n) AS labels, properties(n) AS props
    """
    node_results = graph_store.structured_query(node_query)
    G = nx.Graph()
    id_map = {}

    if isinstance(node_results, list):
        for r in node_results:
            if isinstance(r, dict):
                nid = r["id"]
                name = r.get("name", "")
                labels = r.get("labels", [])
                props = r.get("props", {})
            elif hasattr(r, "values"):
                vals = list(r.values())
                nid, name = vals[0], vals[1]
                labels = vals[2] if len(vals) > 2 else []
                props = vals[3] if len(vals) > 3 else {}
            else:
                continue
            # Remove keys that are explicitly passed to add_node to avoid "multiple values" TypeError
            clean_props = {k: v for k, v in props.items() if k not in ("name", "labels", "id")}
            G.add_node(nid, name=name, labels=labels, **clean_props)
            id_map[nid] = name

    print(f"     {G.number_of_nodes()} nodes loaded.")

    print("  -> Fetching relationships from Neo4j...")
    rel_query = """
    MATCH (a)-[r]->(b)
    WHERE a.name IS NOT NULL AND b.name IS NOT NULL
    RETURN elementId(a) AS src, elementId(b) AS tgt, type(r) AS rel_type
    """
    rel_results = graph_store.structured_query(rel_query)
    if isinstance(rel_results, list):
        for r in rel_results:
            if isinstance(r, dict):
                src, tgt, rtype = r["src"], r["tgt"], r.get("rel_type", "RELATED")
            elif hasattr(r, "values"):
                vals = list(r.values())
                src, tgt, rtype = vals[0], vals[1], vals[2] if len(vals) > 2 else "RELATED"
            else:
                continue
            if src in id_map and tgt in id_map:
                G.add_edge(src, tgt, rel_type=rtype)

    print(f"     {G.number_of_edges()} edges loaded.")
    return G


def detect_communities(G: nx.Graph, resolution: float = 1.0) -> dict:
    """
    Run Louvain community detection on the graph.
    Returns a dict mapping node_id -> community_id.
    """
    print(f"  -> Running Louvain community detection (resolution={resolution})...")
    try:
        communities = nx.community.louvain_communities(G, resolution=resolution, seed=42)
    except AttributeError:
        # Older networkx versions
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G))

    node_to_community = {}
    for cid, members in enumerate(communities):
        for node_id in members:
            node_to_community[node_id] = cid

    print(f"     Detected {len(communities)} communities across {len(node_to_community)} nodes.")

    # Print community size distribution
    sizes = [len(c) for c in communities]
    sizes.sort(reverse=True)
    top_5 = sizes[:5]
    print(f"     Top 5 community sizes: {top_5}")

    return node_to_community


def write_communities_to_neo4j(
    graph_store: Neo4jPropertyGraphStore,
    node_to_community: dict,
):
    """
    Stamp `community_id` on each node in Neo4j.
    """
    print("  -> Writing community_id to Neo4j nodes...")
    # Group by community for batch updates
    nodes_by_comm = defaultdict(list)
    for nid, cid in node_to_community.items():
        nodes_by_comm[cid].append(nid)

    for cid, node_ids in nodes_by_comm.items():
        # Batch update all nodes in this community
        query = """
        UNWIND $node_ids AS nid
        MATCH (n) WHERE elementId(n) = nid
        SET n.community_id = $cid
        """
        try:
            graph_store.structured_query(
                query, param_map={"node_ids": node_ids, "cid": cid}
            )
        except Exception as e:
            print(f"     Warning: Failed to set community_id={cid} for {len(node_ids)} nodes: {e}")

    print(f"     Stamped community_id on {len(node_to_community)} nodes.")


def summarize_communities(
    graph_store: Neo4jPropertyGraphStore,
    G: nx.Graph,
    node_to_community: dict,
):
    """
    Generate LLM summaries for each community and store them as
    (:CommunitySummary) nodes in Neo4j.
    """
    summarizer = CommunitySummarizer(graph_store)

    # Group by community
    nodes_by_comm = defaultdict(list)
    for nid, cid in node_to_community.items():
        nodes_by_comm[cid].append(nid)

    print(f"  -> Summarizing {len(nodes_by_comm)} communities...")

    for cid, node_ids in nodes_by_comm.items():
        # Skip tiny communities (< 3 nodes)
        if len(node_ids) < 3:
            continue

        # Collect entity names and relationships for this community
        entity_names = []
        relationships = []
        for nid in node_ids:
            if nid in G.nodes:
                name = G.nodes[nid].get("name", nid)
                entity_names.append(name)
                for _, neighbor, data in G.edges(nid, data=True):
                    if neighbor in node_ids:
                        neighbor_name = G.nodes[neighbor].get("name", neighbor)
                        rel_type = data.get("rel_type", "RELATED_TO")
                        relationships.append(f"{name} --[{rel_type}]--> {neighbor_name}")

        summarizer.summarize_and_store(
            community_id=cid,
            entity_names=entity_names,
            relationships=relationships,
        )

    print("  -> Community summarization complete.")
