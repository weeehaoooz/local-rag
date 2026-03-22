import os
from dotenv import load_dotenv
from llama_index.core import KnowledgeGraphIndex, VectorStoreIndex, SummaryIndex, StorageContext, Settings, load_index_from_storage
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Load environment variables
load_dotenv()

# Setup Local Models
llm = Ollama(
    model="llama3", 
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    request_timeout=300.0
)
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
)

# Configure global settings
Settings.llm = llm
Settings.embed_model = embed_model

# Setup Neo4j Connection
graph_store = Neo4jGraphStore(
    username=os.getenv("NEO4J_USERNAME", "neo4j"), 
    password=os.getenv("NEO4J_PASSWORD", "password"), 
    url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    refresh_schema=False
)
storage_context = StorageContext.from_defaults(graph_store=graph_store)

class HybridEngine:
    def __init__(self):
        # 1. Connect to existing Graph
        self.kg_index = KnowledgeGraphIndex([], storage_context=storage_context)
        
        # 2. Load Vector and Summary Indexes from storage
        print("Loading indexes from ./storage...")
        
        # Load Vector Index
        vector_storage_context = StorageContext.from_defaults(persist_dir="./storage/vector")
        self.vector_index = load_index_from_storage(vector_storage_context)
        
        # Load Summary Index
        summary_storage_context = StorageContext.from_defaults(persist_dir="./storage/summary")
        self.summary_index = load_index_from_storage(summary_storage_context)

    def get_context(self, query):
        # 1. Ask GraphRAG for connections (e.g. "Who is linked to Project X?")
        graph_nodes = self.kg_index.as_retriever().retrieve(query)
        
        # 2. Ask Vector Index for semantic retrieval
        vector_nodes = self.vector_index.as_retriever().retrieve(query)
        
        # 3. Ask Summary Index for high-level overview or structural info
        summary_nodes = self.summary_index.as_retriever().retrieve(query)
        
        return {
            "graph_context": [n.text for n in graph_nodes],
            "vector_context": [n.text for n in vector_nodes],
            "summary_context": [n.text for n in summary_nodes]
        }
