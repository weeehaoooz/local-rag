import os
import sys
import argparse
import asyncio
from collections import defaultdict
from dotenv import load_dotenv

# Add backend to path
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BACKEND_DIR)

from config import setup_indexing_env
from ingestion.loader import discover_files, SmartDocumentLoader
from ingestion.preprocessor import DocumentPreprocessor
from indexing.tracker import IndexingTracker
from indexing.vector import VectorIndexer
from indexing.bm25 import BM25Indexer
from indexing.summary import SummaryIndexer
from indexing.graph_indexer import GraphIndexer
from indexing.progress import create_progress_handler
from retrieval.guardrails import GuardrailManager, _derive_category, _derive_title
from llama_index.core import StorageContext

load_dotenv()

def summarize_and_guardrail(files_by_category, guardrail_mgr, loader, preprocessor, progress_handler, force=False):
    """Generate per-file summaries and category guardrails."""
    print(f"Processing {len(files_by_category)} category(ies)...")
    
    for category, cat_files in files_by_category.items():
        existing_guardrails = guardrail_mgr.get_guardrails(category)
        if existing_guardrails is not None and not force:
            # Ensure summaries exist even if guardrails do
            for f in cat_files[:5]:
                docs = loader.load(f)
                docs = preprocessor.preprocess(docs)
                guardrail_mgr.ensure_document_summary(f, docs, force=False)
            continue

        print(f"  -> Building guardrails for '{category}'...")
        category_summaries = []
        sample_files = cat_files[:5]
        for f in sample_files:
            docs = loader.load(f)
            docs = preprocessor.preprocess(docs)
            summary = guardrail_mgr.ensure_document_summary(f, docs, force=force)
            category_summaries.append(summary)
            progress_handler.update(sample_files.index(f) + 1, len(sample_files), f"Summarized: {os.path.basename(f)}")
        progress_handler.end()

        sample_docs = loader.load(sample_files[0]) if sample_files else []
        for f in sample_files[1:]:
            sample_docs.extend(loader.load(f))
            
        guardrail_mgr.generate_guardrails(category, sample_docs, summaries=category_summaries)
        print(f"  -> Optimizing guardrails for '{category}'...")
        guardrail_mgr.optimize_guardrails(category)

def index_files(files_to_index, is_regen, indexers, managers, options):
    """Core indexing logic for a set of files."""
    from config import DATA_DIR
    loader, preprocessor, guardrail_mgr, tracker, progress_handler = managers
    vector_indexer, bm25_indexer, summary_indexer, graph_indexer = indexers
    force, hybrid, agentic_chunk, num_passes = options

    if not files_to_index:
        return

    label = "Re-indexing KG" if is_regen else "Indexing"
    print(f"\n{label} {len(files_to_index)} file(s)...")

    files_by_cat = defaultdict(list)
    for f in files_to_index:
        files_by_cat[_derive_category(f, DATA_DIR)].append(f)

    for category, files in files_by_cat.items():
        print(f"  -> Category: [{category}] ({len(files)} files)")
        progress_handler.clear()

        similar_cats = guardrail_mgr.get_similar_categories(category)
        guardrails = guardrail_mgr.get_guardrails(category)

        for idx, f in enumerate(files, 1):
            print(f"    -> {f}")
            documents = loader.load(f)
            documents = preprocessor.preprocess(documents)
            doc_summary = guardrail_mgr.ensure_document_summary(f, documents, force=force)
            
            title = _derive_title(f)
            kg_prefix = guardrail_mgr.build_kg_prompt_prefix(category, title, document_summary=doc_summary)

            for doc in documents:
                doc.metadata.update({"title": title, "category": category, "summary": doc_summary})
                doc.set_content(f"Document Title: {title}\n\n{doc.get_content()}")

            if not is_regen:
                vector_indexer.index_documents(documents)
                bm25_indexer.index_documents(documents, title=title)
                summary_indexer.index_documents(documents)
            
            graph_indexer.index_documents(
                documents,
                max_triplets_per_chunk=10,
                category=category,
                num_passes=num_passes,
                similar_categories=similar_cats,
                guardrails=guardrails,
                title=title,
                kg_prompt_prefix=kg_prefix,
                include_free_form=hybrid,
                agentic_chunk=agentic_chunk,
            )
            if not is_regen:
                tracker.update_file_hash(f)
            
            progress_handler.update(idx, len(files), f"Processed: {os.path.basename(f)}")
        progress_handler.end()


async def run_pipeline(step="all", force=False, hybrid=False, agentic_chunk=False, passes=1, **kwargs):
    print(f"\n🚀 Running Pipeline Step: {step.upper()} (Force: {force})")
    from config import (
        DATA_DIR, STORAGE_DIR, INDEXING_STATE, 
        VECTOR_DIR, BM25_DIR, SUMMARY_DIR
    )
    
    # 1. Setup
    graph_store = setup_indexing_env()
    storage_context = StorageContext.from_defaults(property_graph_store=graph_store)

    tracker = IndexingTracker(INDEXING_STATE)
    guardrail_mgr = GuardrailManager()
    loader = SmartDocumentLoader(enable_vision=kwargs.get("vision", False))
    preprocessor = DocumentPreprocessor(enable_llm_coref=kwargs.get("llm_coref", False))
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

    # --- STEP: GraphRAG Community Detection ---
    if step in ["index", "all", "graph-rag"] and kwargs.get("graph_rag"):
        print("\n--- [OPTIONAL STEP] GRAPHRAG COMMUNITY DETECTION ---")
        # Ensure indexers are initialized if we only run this step
        if 'graph_indexer' not in locals():
            graph_indexer = GraphIndexer(storage_context)
        if 'vector_indexer' not in locals():
            vector_indexer = VectorIndexer()
            vector_indexer.load(VECTOR_DIR)
            
        community_docs = graph_indexer.detect_and_summarize_communities()
        if community_docs:
            print(f"  -> Indexing {len(community_docs)} community summaries into Vector store...")
            vector_indexer.index_documents(community_docs)
            vector_indexer.persist(VECTOR_DIR)
            print("✅ GraphRAG community detection and indexing complete.")
        else:
            print("  -> No communities detected or summarized.")

    graph_store.close()
    print("\n✨ Pipeline run finished successfully.")

def main():
    parser = argparse.ArgumentParser(description="Trigger specific RAG pipeline steps.")
    parser.add_argument("--step", choices=["ingest", "summarize", "index", "all"], default="all", help="Pipeline step to run")
    parser.add_argument("--force", action="store_true", help="Force execution even if no changes detected")
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid KG extraction (Schema + Free-form)")
    parser.add_argument("--agentic-chunk", action="store_true", help="Enable agentic chunking (LLM-guided)")
    parser.add_argument("--passes", type=int, default=1, help="Number of graph extraction passes")
    parser.add_argument("--graph-rag", action="store_true", help="Enable GraphRAG community detection and summarization")
    parser.add_argument("--llm-coref", action="store_true", help="Enable high-quality LLM-based coreference resolution")
    parser.add_argument("--vision", action="store_true", help="Enable vision-based PDF metadata extraction")
    
    args = parser.parse_args()
    
    # Run async
    asyncio.run(run_pipeline(
        step=args.step, 
        force=args.force, 
        hybrid=args.hybrid, 
        agentic_chunk=args.agentic_chunk, 
        passes=args.passes,
        graph_rag=args.graph_rag,
        llm_coref=args.llm_coref,
        vision=args.vision
    ))

if __name__ == "__main__":
    main()
