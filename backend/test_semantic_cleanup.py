import os
from llama_index.core import Settings
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.embeddings.ollama import OllamaEmbedding
from indexers.graph import GraphCleaner
from dotenv import load_dotenv
from typing import Any

load_dotenv()

def test_cleanup():
    embed_model = OllamaEmbedding(
        model_name="nomic-embed-text", 
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    Settings.embed_model = embed_model
    
    graph_store = Neo4jPropertyGraphStore(
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
        url=os.getenv("NEO4J_URI", "bolt://localhost:7687")
    )
    
    # 1. Clean up any existing test nodes
    graph_store.structured_query("MATCH (p:Person) WHERE p.name IN ['Artificial Intelligence', 'AI'] DETACH DELETE p")
    
    # 2. Insert two synonymous nodes manually via Cypher
    print("Inserting synonymous nodes...")
    graph_store.structured_query("CREATE (p1:Person {name: 'Artificial Intelligence', entity_type: 'Person', category: 'test_clean'})")
    graph_store.structured_query("CREATE (p2:Person {name: 'AI', entity_type: 'Person', category: 'test_clean'})")
    
    # Add a relationship to each to see if they merge
    graph_store.structured_query("MATCH (p:Person {name: 'Artificial Intelligence'}) MERGE (p)-[:WORKS_IN]->(:Field {name: 'Computer Science'})")
    graph_store.structured_query("MATCH (p:Person {name: 'AI'}) MERGE (p)-[:WORKS_IN]->(:Field {name: 'Computer Science'})")
    
    print("Running semantic cleanup...")
    cleaner = GraphCleaner(graph_store, embed_model=embed_model)
    # 'AI' and 'Artificial Intelligence' are very similar in embedding space
    cleaner.run_cleanup(similarity_threshold=0.8)
    
    # 3. Verify if they were merged
    res = graph_store.structured_query("MATCH (p:Person) WHERE p.name IN ['AI', 'Artificial Intelligence'] RETURN count(p) as count")
    print(f"Nodes remaining (expected 1): {res}")
    
    graph_store.close()

if __name__ == "__main__":
    test_cleanup()
