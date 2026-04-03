import os

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
