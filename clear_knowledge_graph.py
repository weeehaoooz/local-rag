import os
import sys
from dotenv import load_dotenv
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from pageindex import PageIndexClient, PageIndexAPIError
from tqdm import tqdm
import shutil

# Load environment variables
load_dotenv()

def clear_neo4j():
    """Clear all nodes and relationships from Neo4j."""
    print("\n--- Clearing Neo4j Knowledge Graph ---")
    try:
        graph_store = Neo4jGraphStore(
            username=os.getenv("NEO4J_USERNAME", "neo4j"), 
            password=os.getenv("NEO4J_PASSWORD", "password"), 
            url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            refresh_schema=False
        )
        
        print("Executing Cypher: MATCH (n) DETACH DELETE n")
        # DETACH DELETE removes nodes and all their relationships
        result = graph_store.query("MATCH (n) DETACH DELETE n")
        print("Neo4j cleared successfully.")
    except Exception as e:
        print(f"Error clearing Neo4j: {e}")

def clear_pageindex():
    """Delete all documents from PageIndex."""
    print("\n--- Clearing PageIndex Structural Index ---")
    api_key = os.getenv("PAGEINDEX_API_KEY")
    
    if not api_key:
        print("Warning: PAGEINDEX_API_KEY not found in environment or .env file.")
        print("Skipping PageIndex clearing. Please provide an API key to clear PageIndex.")
        return

    try:
        pi_client = PageIndexClient(api_key=api_key)
        
        # List all documents
        print("Fetching document list from PageIndex...")
        response = pi_client.list_documents(limit=100)
        documents = response.get("documents", [])
        total = response.get("total", 0)
        
        if not documents:
            print("No documents found in PageIndex.")
            return

        print(f"Found {total} documents to delete.")
        
        # Delete each document
        for doc in tqdm(documents, desc="Deleting PageIndex documents"):
            doc_id = doc.get("id")
            doc_name = doc.get("name", "Unknown")
            try:
                pi_client.delete_document(doc_id)
                # tqdm.write(f" -> Deleted: {doc_name} ({doc_id})")
            except PageIndexAPIError as e:
                tqdm.write(f" -> Failed to delete {doc_name}: {e}")
                
        print("PageIndex cleared successfully.")
    except Exception as e:
        print(f"Error clearing PageIndex: {e}")

def clear_local_memory():
    """Clear local indexing state and related storage."""
    print("\n--- Clearing Local Indexing Memory ---")
    
    storage_path = "./storage"
    indexing_state = os.path.join(storage_path, "indexing_state.json")
    vector_path = os.path.join(storage_path, "vector")
    summary_path = os.path.join(storage_path, "summary")
    guardrails_path = "./indexers/generated_guardrails"

    # 1. Clear indexing state file
    if os.path.exists(indexing_state):
        try:
            os.remove(indexing_state)
            print(f"Deleted {indexing_state}")
        except Exception as e:
            print(f"Error deleting {indexing_state}: {e}")
    else:
        print(f"{indexing_state} not found, skipping.")

    # 2. Clear vector storage
    if os.path.exists(vector_path):
        try:
            shutil.rmtree(vector_path)
            os.makedirs(vector_path)
            print(f"Cleared {vector_path}")
        except Exception as e:
            print(f"Error clearing {vector_path}: {e}")

    # 3. Clear summary storage
    if os.path.exists(summary_path):
        try:
            shutil.rmtree(summary_path)
            os.makedirs(summary_path)
            print(f"Cleared {summary_path}")
        except Exception as e:
            print(f"Error clearing {summary_path}: {e}")

    # 4. Clear generated guardrails
    if os.path.exists(guardrails_path):
        try:
            # We don't want to delete the directory itself, only contents
            for filename in os.listdir(guardrails_path):
                file_path = os.path.join(guardrails_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
            print(f"Cleared contents of {guardrails_path}")
        except Exception as e:
            print(f"Error clearing {guardrails_path}: {e}")
    
    print("Local memory cleared successfully.")

if __name__ == "__main__":
    confirm = input("Are you sure you want to clear your ENTIRE knowledge graph? (y/N): ")
    if confirm.lower() == 'y':
        clear_neo4j()
        clear_pageindex()
        clear_local_memory()
        print("\nKnowledge graph and local memory clearing complete.")
    else:
        print("Operation cancelled.")
