import os
import sys

# Add backend to path for imports
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# ── Python 3.14 / sniffio compatibility ───────────────────────────────
import sniffio_compat
sniffio_compat.apply()
# ──────────────────────────────────────────────────────────────────────

import httpx
from dotenv import load_dotenv
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from config import DEFAULT_LLM, DEFAULT_EMBED

# Load environment variables
load_dotenv()

def check_ollama():
    """Check if Ollama service is reachable and has the required models."""
    print("--- Checking Ollama Service ---")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    required_models = [DEFAULT_LLM, DEFAULT_EMBED]

    try:
        response = httpx.get(f"{base_url}/api/tags")
        if response.status_code == 200:
            print(f"✅ Ollama service is reachable at {base_url}")
            models = [m["name"] for m in response.json().get("models", [])]
            for model in required_models:
                if model in models:
                    print(f"✅ Model '{model}' found.")
                elif any(model in m for m in models):
                    # Handle version tag variations
                    print(f"✅ Model similar to '{model}' found.")
                else:
                    print(f"❌ Model '{model}' NOT found. Please run: ollama pull {model}")
        else:
            print(f"❌ Ollama service responded with status {response.status_code}")
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")

def check_neo4j():
    """Check if Neo4j service is reachable and basic stats."""
    print("\n--- Checking Neo4j Service ---")
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")

    try:
        store = Neo4jGraphStore(
            username=username,
            password=password,
            url=uri,
            refresh_schema=False
        )
        # Attempt a simple query
        nodes_res = store.query("MATCH (n) RETURN count(n) as count")
        print(f"✅ Neo4j service is reachable at {uri}")
        print(f"   Nodes in Graph: {nodes_res}")
        
        rel_res = store.query("MATCH ()-[r]->() RETURN count(r) as count")
        print(f"   Relationships in Graph: {rel_res}")
    except Exception as e:
        print(f"❌ Error connecting to Neo4j: {e}")

def check_storage():
    """Check local storage directories."""
    print("\n--- Checking Local Storage ---")
    from config import STORAGE_DIR
    storage_path = STORAGE_DIR
    subdirs = ["vector", "bm25", "summary", "generated_guardrails", "generated_summaries"]

    if os.path.exists(storage_path):
        print(f"✅ Storage root found at {storage_path}")
        for subdir in subdirs:
            p = os.path.join(storage_path, subdir)
            if os.path.exists(p):
                print(f"✅ Subdirectory '{subdir}' exists.")
            else:
                print(f"⚠️ Subdirectory '{subdir}' NOT found. It will be created during indexing.")
    else:
        print(f"❌ Storage root NOT found at {storage_path}. Creating it now...")
        os.makedirs(storage_path, exist_ok=True)
        print(f"✅ Storage root created.")

def main():
    check_ollama()
    check_neo4j()
    check_storage()

if __name__ == "__main__":
    main()
