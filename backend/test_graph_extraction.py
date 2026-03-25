import os
import asyncio
import nest_asyncio
from dotenv import load_dotenv
from llama_index.core import StorageContext, Settings
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from indexers.graph import GraphIndexer

# Apply nest_asyncio
nest_asyncio.apply()
load_dotenv()

async def test_extraction():
    # 1. Setup
    llm = Ollama(model="llama3:latest", base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    graph_store = Neo4jPropertyGraphStore(
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
        url=os.getenv("NEO4J_URI", "bolt://localhost:7687")
    )
    storage_context = StorageContext.from_defaults(property_graph_store=graph_store)
    
    indexer = GraphIndexer(storage_context)
    
    # 2. Create a simple node
    node = TextNode(text="Marcus Lim is a Software Engineer at Google.", id_="test_node_1")
    
    # 3. Define dummy guardrails
    guardrails = {
        "business_objects": [
            {"name": "Person", "properties": ["name"]},
            {"name": "Company", "properties": ["name"]}
        ],
        "relationship_types": ["WORKS_FOR"],
        "conventions": "Be concise."
    }
    
    print("Starting indexing...")
    indexer.index_documents(
        [node],
        guardrails=guardrails,
        category="test",
        title="Test Doc"
    )
    print("Indexing complete.")
    
    # 4. Verify in Neo4j
    query = "MATCH (n:Person {name: 'Marcus Lim'})-[r:WORKS_FOR]->(c:Company {name: 'Google'}) RETURN n, r, c"
    res = graph_store.query(query)
    print(f"Verification Query Result: {res}")
    
    graph_store.close()

if __name__ == "__main__":
    asyncio.run(test_extraction())
