import os
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

# Project root is two levels up from backend/config.py
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

# Centralized storage directory at project root
STORAGE_DIR = os.path.join(PROJECT_ROOT, "storage")
DATA_DIR = os.path.join(BACKEND_DIR, "data")

# Specific subdirectories (centralized names)
VECTOR_DIR = os.path.join(STORAGE_DIR, "vector")
BM25_DIR = os.path.join(STORAGE_DIR, "bm25")
SUMMARY_DIR = os.path.join(STORAGE_DIR, "summary")
GUARDRAILS_DIR = os.path.join(STORAGE_DIR, "generated_guardrails")
SUMMARIES_DIR = os.path.join(STORAGE_DIR, "generated_summaries")
INDEXING_STATE = os.path.join(STORAGE_DIR, "indexing_state.json")

# Ensure base storage exists
os.makedirs(STORAGE_DIR, exist_ok=True)

def setup_indexing_env():
    """Setup models and global settings."""
    llm = Ollama(
        model="gemma4:latest",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        request_timeout=720.0,
        context_window=int(os.getenv("OLLAMA_CONTEXT_WINDOW", "8192")),
        additional_kwargs={
            "num_ctx": int(os.getenv("OLLAMA_CONTEXT_WINDOW", "8192")),
        },
    )
    embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        request_timeout=360.0,
    )

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.num_workers = 1

    graph_store = Neo4jPropertyGraphStore(
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
        url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        refresh_schema=False,
    )
    return graph_store
