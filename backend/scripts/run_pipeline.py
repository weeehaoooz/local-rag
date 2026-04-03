import os
import sys
import argparse
import asyncio
from dotenv import load_dotenv

# Add backend to path
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BACKEND_DIR)

from indexer import (
    setup_indexing_env, discover_files, summarize_and_guardrail, 
    index_files, IndexingTracker, GuardrailManager, 
    SmartDocumentLoader, DocumentPreprocessor, create_progress_handler,
    VectorIndexer, BM25Indexer, SummaryIndexer, GraphIndexer, StorageContext
)

load_dotenv()

async def run_pipeline(step="all", force=False, hybrid=False, agentic_chunk=False, passes=1):
    print(f"\n🚀 Running Pipeline Step: {step.upper()} (Force: {force})")
    from config import STORAGE_DIR, INDEXING_STATE, VECTOR_DIR, BM25_DIR, SUMMARY_DIR
    
    # 1. Setup
    graph_store = setup_indexing_env()
    storage_context = StorageContext.from_defaults(property_graph_store=graph_store)

    tracker = IndexingTracker(INDEXING_STATE)
    guardrail_mgr = GuardrailManager()
    loader = SmartDocumentLoader()
    preprocessor = DocumentPreprocessor()
    progress_handler = create_progress_handler()

    # 2. Discovery (Always needed for context)
    files_by_category, all_files = discover_files()
    if not all_files:
        print("❌ No files found in ./data")
        graph_store.close()
        return

    # --- STEP: INGEST ---
    if step in ["ingest", "all"]:
        print("\n--- [STEP 1/3] INGESTION ---")
        # Ingestion is implicitly handled by discovery and loader
        # We just verify files are readable
        print(f"✅ Discovered {len(all_files)} files across {len(files_by_category)} categories.")
        if step == "ingest":
            graph_store.close()
            return

    # --- STEP: SUMMARIZE ---
    if step in ["summarize", "all"]:
        print("\n--- [STEP 2/3] SUMMARIZATION & GUARDRAILS ---")
        summarize_and_guardrail(files_by_category, guardrail_mgr, loader, preprocessor, progress_handler, force=force)
        print("✅ Summarization & Guardrails complete.")
        if step == "summarize":
            graph_store.close()
            return

    # --- STEP: INDEX ---
    if step in ["index", "all"]:
        print("\n--- [STEP 3/3] INDEXING (Vector, BM25, Graph) ---")
        # Delta Detection
        dirty_files = [f for f in all_files if force or tracker.is_file_changed(f)]
        kg_regen_files = []
        for category in files_by_category:
            gr_hash = guardrail_mgr.guardrails_hash(category)
            if gr_hash and (force or tracker.is_guardrail_changed(category, gr_hash)):
                for f in files_by_category[category]:
                    if f not in dirty_files:
                        kg_regen_files.append(f)

        if not dirty_files and not kg_regen_files:
            print("✅ All indexes are already up to date.")
        else:
            # Initialize Indexers
            vector_indexer = VectorIndexer()
            bm25_indexer = BM25Indexer()
            summary_indexer = SummaryIndexer()
            graph_indexer = GraphIndexer(storage_context)

            vector_indexer.load(VECTOR_DIR)
            bm25_indexer.load(BM25_DIR)
            summary_indexer.load(SUMMARY_DIR)

            indexers = (vector_indexer, bm25_indexer, summary_indexer, graph_indexer)
            managers = (loader, preprocessor, guardrail_mgr, tracker, progress_handler)
            options = (force, hybrid, agentic_chunk, passes)

            # Perform indexing
            if dirty_files:
                index_files(dirty_files, False, indexers, managers, options)
            if kg_regen_files:
                index_files(kg_regen_files, True, indexers, managers, options)

            # Persist and Cleanup
            for category in files_by_category:
                gr_hash = guardrail_mgr.guardrails_hash(category)
                if gr_hash:
                    tracker.update_guardrail_hash(category, gr_hash)

            vector_indexer.persist(VECTOR_DIR)
            bm25_indexer.persist(BM25_DIR)
            summary_indexer.persist(SUMMARY_DIR)
            tracker.save_state()
            graph_indexer.clear_cache()
            print("✅ Indexing complete.")

    graph_store.close()
    print("\n✨ Pipeline run finished successfully.")

def main():
    parser = argparse.ArgumentParser(description="Trigger specific RAG pipeline steps.")
    parser.add_argument("--step", choices=["ingest", "summarize", "index", "all"], default="all", help="Pipeline step to run")
    parser.add_argument("--force", action="store_true", help="Force execution even if no changes detected")
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid KG extraction (Schema + Free-form)")
    parser.add_argument("--agentic-chunk", action="store_true", help="Enable agentic chunking (LLM-guided)")
    parser.add_argument("--passes", type=int, default=1, help="Number of graph extraction passes")
    
    args = parser.parse_args()
    
    # Run async
    asyncio.run(run_pipeline(
        step=args.step, 
        force=args.force, 
        hybrid=args.hybrid, 
        agentic_chunk=args.agentic_chunk, 
        passes=args.passes
    ))

if __name__ == "__main__":
    main()
