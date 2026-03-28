"""
scripts/detect_communities.py
------------------------------
Detects communities in the Neo4j Knowledge Graph using the Louvain algorithm
via NetworkX and writes `community_id` + `community_level` back to Neo4j.

Usage:
    python scripts/detect_communities.py [--resolution 1.0] [--summarize]

Flags:
    --resolution   Louvain resolution parameter (higher = more communities)
    --summarize    After detection, run LLM community summarization
"""

import os
import sys
import argparse
from collections import defaultdict
from dotenv import load_dotenv

# Ensure backend root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import networkx as nx
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


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
            G.add_node(nid, name=name, labels=labels, **props)
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
    community_to_nodes = defaultdict(list)
    for nid, cid in node_to_community.items():
        community_to_nodes[cid].append(nid)

    for cid, node_ids in community_to_nodes.items():
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
    from indexers.community import CommunitySummarizer

    summarizer = CommunitySummarizer(graph_store)

    # Group by community
    community_to_nodes = defaultdict(list)
    for nid, cid in node_to_community.items():
        community_to_nodes[cid].append(nid)

    print(f"  -> Summarizing {len(community_to_nodes)} communities...")

    for cid, node_ids in community_to_nodes.items():
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


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Detect and summarize communities in the Knowledge Graph.")
    parser.add_argument("--resolution", type=float, default=1.0, help="Louvain resolution (higher = more communities)")
    parser.add_argument("--summarize", action="store_true", help="Generate LLM summaries for each community")
    args = parser.parse_args()

    # Setup models
    Settings.llm = Ollama(
        model="llama3:latest",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        request_timeout=720.0,
        context_window=int(os.getenv("OLLAMA_CONTEXT_WINDOW", "8192")),
        additional_kwargs={"num_ctx": int(os.getenv("OLLAMA_CONTEXT_WINDOW", "8192"))},
    )
    Settings.embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        request_timeout=360.0,
    )

    print("Connecting to Neo4j...")
    graph_store = Neo4jPropertyGraphStore(
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
        url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        refresh_schema=False,
    )

    print("\n--- Community Detection ---")
    G = build_networkx_graph(graph_store)

    if G.number_of_nodes() < 2:
        print("Graph has fewer than 2 nodes. Nothing to cluster.")
        return

    node_to_community = detect_communities(G, resolution=args.resolution)
    write_communities_to_neo4j(graph_store, node_to_community)

    if args.summarize:
        print("\n--- Community Summarization ---")
        summarize_communities(graph_store, G, node_to_community)

    print("\n--- Done ---")
    graph_store.close()
    Settings.llm = None
    Settings.embed_model = None


if __name__ == "__main__":
    main()
