import os
import sys
import argparse
import shutil
from dotenv import load_dotenv
from collections import defaultdict

# Add backend to path for imports
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# ── Python 3.14 / sniffio compatibility ───────────────────────────────
import sniffio_compat
sniffio_compat.apply()
# ──────────────────────────────────────────────────────────────────────

from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core import StorageContext, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from indexing.graph_indexer import GraphIndexer
from config import DEFAULT_LLM, DEFAULT_EMBED

# Load environment variables
load_dotenv()

def setup_models():
    """Configure LLM and Embedding models for management tasks."""
    Settings.llm = Ollama(
        model=DEFAULT_LLM,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        request_timeout=720.0,
        context_window=int(os.getenv("OLLAMA_CONTEXT_WINDOW", "8192")),
        additional_kwargs={"num_ctx": int(os.getenv("OLLAMA_CONTEXT_WINDOW", "8192"))},
    )
    Settings.embed_model = OllamaEmbedding(
        model_name=DEFAULT_EMBED,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        request_timeout=360.0,
    )

def get_graph_store():
    """Connect to Neo4j and return the graph store."""
    return Neo4jPropertyGraphStore(
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
        url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        refresh_schema=False,
    )

def clear_indexes(args):
    """Clear all indexes and state."""
    print("\n--- Clearing All Indexes & State ---")
    
    # 1. Clear Neo4j
    try:
        store = get_graph_store()
        print("Clearing Neo4j: MATCH (n) DETACH DELETE n")
        store.structured_query("MATCH (n) DETACH DELETE n")
        print("✅ Neo4j cleared.")
    except Exception as e:
        print(f"❌ Error clearing Neo4j: {e}")

    # 2. Clear Local Storage
    from config import STORAGE_DIR
    storage_path = STORAGE_DIR
    items_to_clear = [
        ("indexing_state.json", "file"),
        ("vector", "dir"),
        ("bm25", "dir"),
        ("summary", "dir"),
        ("generated_guardrails", "dir"),
        ("generated_summaries", "dir"),
    ]

    for name, item_type in items_to_clear:
        path = os.path.join(storage_path, name)
        if os.path.exists(path):
            try:
                if item_type == "file":
                    os.remove(path)
                else:
                    for filename in os.listdir(path):
                        file_path = os.path.join(path, filename)
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                print(f"✅ Cleared {name}")
            except Exception as e:
                print(f"❌ Error clearing {name}: {e}")
        else:
            print(f"ℹ️ {name} not found, skipping.")

    print("\nFull reset complete.")

def clean_indexes(args):
    """Clean Knowledge Graph by merging similar entries."""
    print("\n--- Cleaning Knowledge Graph (Semantic Cleanup) ---")
    setup_models()
    
    try:
        graph_store = get_graph_store()
        storage_context = StorageContext.from_defaults(property_graph_store=graph_store)
        indexer = GraphIndexer(storage_context)
        
        print(f"Running semantic cleanup (threshold={args.threshold})...")
        indexer.clean_graph(similarity_threshold=args.threshold, rel_threshold=args.rel_threshold)
        print("✅ Graph cleanup complete.")
    except Exception as e:
        print(f"❌ Error during graph cleanup: {e}")

def cluster_indexes(args):
    """Run community detection and summarization."""
    print("\n--- Running Community Detection & Summarization ---")
    from indexing.community import build_networkx_graph, detect_communities, write_communities_to_neo4j, summarize_communities
    
    setup_models()
    
    try:
        graph_store = get_graph_store()
        G = build_networkx_graph(graph_store)

        if G.number_of_nodes() < 2:
            print("Graph has fewer than 2 nodes. Nothing to cluster.")
            return

        node_to_community = detect_communities(G, resolution=args.resolution)
        write_communities_to_neo4j(graph_store, node_to_community)

        if args.summarize:
            print("\n--- Generating Community Summaries ---")
            summarize_communities(graph_store, G, node_to_community)
        
        print("✅ Clustering complete.")
    except Exception as e:
        print(f"❌ Error during clustering: {e}")

def show_stats(args):
    """Show status and statistics of all indexes."""
    print("\n--- Index Statistics ---")
    
    # 1. Neo4j Stats
    try:
        store = get_graph_store()
        nodes = store.structured_query("MATCH (n) RETURN count(n) as count")
        rels = store.structured_query("MATCH ()-[r]->() RETURN count(r) as count")
        communities = store.structured_query("MATCH (n) WHERE n.community_id IS NOT NULL RETURN count(DISTINCT n.community_id) as count")
        
        # Parse results which are lists of dicts
        node_count = nodes[0]["count"] if nodes and "count" in nodes[0] else 0
        rel_count = rels[0]["count"] if rels and "count" in rels[0] else 0
        comm_count = communities[0]["count"] if communities and "count" in communities[0] else 0

        print(f"Neo4j Knowledge Graph:")
        print(f"  Nodes: {node_count}")
        print(f"  Relationships: {rel_count}")
        print(f"  Detected Communities: {comm_count}")
    except Exception as e:
        print(f"❌ Error fetching Neo4j stats: {e}")

    # 2. Disk Stats
    from config import STORAGE_DIR
    storage_path = STORAGE_DIR
    if not os.path.exists(storage_path):
        print(f"\nLocal Storage ({storage_path}): Not found.")
        return

    print(f"\nLocal Storage ({storage_path}):")
    for item in os.listdir(storage_path):
        item_path = os.path.join(storage_path, item)
        if os.path.isdir(item_path):
            size = sum(os.path.getsize(os.path.join(dirpath, filename)) for dirpath, _, filenames in os.walk(item_path) for filename in filenames)
            print(f"  {item: <20} : {size / 1024 / 1024:.2f} MB")
        elif os.path.isfile(item_path):
            size = os.path.getsize(item_path)
            print(f"  {item: <20} : {size / 1024:.2f} KB")

def main():
    parser = argparse.ArgumentParser(description="Manage RAG indexes and knowledge graph.")
    subparsers = parser.add_subparsers(dest="command", help="Management command to run")

    # Clear command
    subparsers.add_parser("clear", help="Clear all indexes, storage, and state")

    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Perform semantic cleanup of the Knowledge Graph")
    clean_parser.add_argument("--threshold", type=float, default=0.9, help="Similarity threshold for merging nodes")
    clean_parser.add_argument("--rel-threshold", type=float, default=None, help="Similarity threshold for merging relationships")

    # Cluster command
    cluster_parser = subparsers.add_parser("cluster", help="Run community detection and summarization")
    cluster_parser.add_argument("--resolution", type=float, default=1.0, help="Louvain resolution parameter")
    cluster_parser.add_argument("--summarize", action="store_true", help="Generate LLM summaries for communities")

    # Stats command
    subparsers.add_parser("stats", help="Show current index statistics")

    args = parser.parse_args()

    if args.command == "clear":
        confirm = input("⚠️ This will DELETE all indexed data. Are you sure? (y/N): ")
        if confirm.lower() == 'y':
            clear_indexes(args)
    elif args.command == "clean":
        clean_indexes(args)
    elif args.command == "cluster":
        cluster_indexes(args)
    elif args.command == "stats":
        show_stats(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
