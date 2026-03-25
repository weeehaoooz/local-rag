import os
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from dotenv import load_dotenv

load_dotenv()

def verify():
    print("--- Verifying Neo4j Content ---")
    store = Neo4jGraphStore(
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
        url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        refresh_schema=False
    )
    
    # 1. Count nodes
    nodes_query = "MATCH (n) RETURN count(n) as count"
    nodes_res = store.query(nodes_query)
    print(f"Total Nodes: {nodes_res}")
    
    # 2. Sample nodes
    sample_query = "MATCH (n) RETURN n.name as name, labels(n) as labels LIMIT 10"
    sample_res = store.query(sample_query)
    print("Sample Nodes:")
    for r in sample_res:
        print(f"  - {r}")
    
    # 3. Count relationships
    rel_query = "MATCH ()-[r]->() RETURN count(r) as count"
    rel_res = store.query(rel_query)
    print(f"Total Relationships: {rel_res}")

if __name__ == "__main__":
    verify()
