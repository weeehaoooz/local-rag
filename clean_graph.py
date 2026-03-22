import os
import argparse
from dotenv import load_dotenv
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core import StorageContext, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from indexers.graph import GraphIndexer

def main():
    load_dotenv()
    
    # Setup Local Models
    Settings.llm = Ollama(
        model="llama3", 
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        request_timeout=300.0
    )
    Settings.embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )

    parser = argparse.ArgumentParser(description="Clean up knowledge graph by merging similar entities and relationships.")
    parser.add_argument("--threshold", "--node-threshold", type=float, default=0.9, help="Similarity threshold (0.0 to 1.0) for merging nodes.")
    parser.add_argument("--rel-threshold", type=float, default=None, help="Similarity threshold (0.0 to 1.0) for merging relationship types. Defaults to --threshold.")
    args = parser.parse_args()

    print(f"Connecting to Neo4j...")
    try:
        graph_store = Neo4jPropertyGraphStore(
            username=os.getenv("NEO4J_USERNAME", "neo4j"), 
            password=os.getenv("NEO4J_PASSWORD", "password"), 
            url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        )
        storage_context = StorageContext.from_defaults(property_graph_store=graph_store)
        
        indexer = GraphIndexer(storage_context)
        indexer.clean_graph(similarity_threshold=args.threshold, rel_threshold=args.rel_threshold)
        
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")

if __name__ == "__main__":
    main()
