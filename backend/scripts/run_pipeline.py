import os
import sys
import argparse
import asyncio
from collections import defaultdict
from dotenv import load_dotenv

# Add backend to path
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# ── Python 3.14 / sniffio compatibility ───────────────────────────────
import sniffio_compat
sniffio_compat.apply()
# ──────────────────────────────────────────────────────────────────────

from config import setup_indexing_env
from ingestion.loader import discover_files, SmartDocumentLoader
from ingestion.preprocessor import DocumentPreprocessor
from indexing.tracker import IndexingTracker
from indexing.vector import VectorIndexer
from indexing.bm25 import BM25Indexer
from indexing.summary import SummaryIndexer
from indexing.graph_indexer import GraphIndexer
from indexing.progress import create_progress_handler
from retrieval.guardrails import GuardrailManager, _derive_title
from llama_index.core import StorageContext

load_dotenv()

import json
from llama_index.core import Settings

async def _assign_categories_via_llm(all_files, loader, preprocessor, guardrail_mgr, progress_handler):
    """Categorize files dynamically using an LLM to group them by semantic content."""
    from config import STORAGE_DIR
    map_path = os.path.join(STORAGE_DIR, "category_map.json")
    
    category_map = {}
    if os.path.exists(map_path):
        try:
            with open(map_path, "r") as f:
                category_map = json.load(f)
        except json.JSONDecodeError:
            pass

    new_files = [f for f in all_files if f not in category_map]
    existing_categories = set(category_map.values())

    if new_files:
        print(f"\n--- DYNAMIC CATEGORIZATION ---")
        print(f"Categorizing {len(new_files)} new file(s) via LLM...")
        progress_handler.clear()

        async def process_file(idx, f):
            # We still blockingly read the file via loader, but that's fast.
            docs = loader.load(f)
            docs = preprocessor.preprocess(docs)
            summary = await guardrail_mgr.aensure_document_summary(f, docs)
            
            prompt = (
                "You are a taxonomy expert categorising documents into logical domains.\n"
                f"Existing Categories: {list(existing_categories) if existing_categories else 'None yet'}\n\n"
                f"Document Summary:\n{summary}\n\n"
                "Task: Assign this document to a category. Return ONLY a single snake_case string representing the category name. "
                "If it strongly fits an existing category, output that exact name. If not, create a succinct new category name."
            )
            try:
                response = await Settings.llm.acomplete(prompt)
                cat = response.text.strip().lower().replace(" ", "_").strip('\"\'')
                cat = ''.join(c for c in cat if c.isalnum() or c == '_')
                if not cat:
                    cat = "general"
            except Exception as e:
                print(f"    [Warning] LLM categorization failed for {os.path.basename(f)}: {e}. Falling back to 'general'.")
                cat = "general"
            
            return f, cat, idx

        tasks = [process_file(idx, f) for idx, f in enumerate(new_files, 1)]
        results = await asyncio.gather(*tasks)

        for f, cat, idx in results:
            category_map[f] = cat
            existing_categories.add(cat)
            progress_handler.update(idx, len(new_files), f"Categorized -> {cat}")
        
        progress_handler.end()
        with open(map_path, "w") as f:
            json.dump(category_map, f, indent=2)

    files_by_category = defaultdict(list)
    for f in all_files:
        files_by_category[category_map.get(f, "general")].append(f)

    return files_by_category, category_map

async def summarize_and_guardrail(files_by_category, guardrail_mgr, loader, preprocessor, progress_handler, force=False):
    """Generate per-file summaries and category guardrails."""
    print(f"Processing {len(files_by_category)} category(ies)...")
    
    for category, cat_files in files_by_category.items():
        existing_guardrails = guardrail_mgr.get_guardrails(category)
        if existing_guardrails is not None and not force:
            # Ensure summaries exist even if guardrails do
            async def load_and_ensure(f):
                docs = loader.load(f)
                docs = preprocessor.preprocess(docs)
                await guardrail_mgr.aensure_document_summary(f, docs, force=False)
            
            await asyncio.gather(*(load_and_ensure(f) for f in cat_files[:5]))
            continue

        print(f"  -> Building guardrails for '{category}'...")
        sample_files = cat_files[:5]
        
        async def load_and_summarize(idx, f):
            docs = loader.load(f)
            docs = preprocessor.preprocess(docs)
            summary = await guardrail_mgr.aensure_document_summary(f, docs, force=force)
            return summary, f, idx
        
        tasks = [load_and_summarize(idx, f) for idx, f in enumerate(sample_files)]
        results = await asyncio.gather(*tasks)
        results = sorted(results, key=lambda x: x[2])
        category_summaries = [r[0] for r in results]
        
        for _, f, _ in results:
            progress_handler.update(sample_files.index(f) + 1, len(sample_files), f"Summarized: {os.path.basename(f)}")
        progress_handler.end()

        sample_docs = loader.load(sample_files[0]) if sample_files else []
        for f in sample_files[1:]:
            sample_docs.extend(loader.load(f))
            
        guardrail_mgr.generate_guardrails(category, sample_docs, summaries=category_summaries)
        print(f"  -> Optimizing guardrails for '{category}'...")
        guardrail_mgr.optimize_guardrails(category)

async def index_files(files_to_index, is_regen, indexers, managers, options, category_map):
    """Core indexing logic for a set of files."""
    loader, preprocessor, guardrail_mgr, tracker, progress_handler = managers
    vector_indexer, bm25_indexer, summary_indexer, graph_indexer = indexers
    force, hybrid, agentic_chunk, num_passes = options

    if not files_to_index:
        return

    label = "Re-indexing KG" if is_regen else "Indexing"
    print(f"\n{label} {len(files_to_index)} file(s)...")

    files_by_cat = defaultdict(list)
    for f in files_to_index:
        files_by_cat[category_map.get(f, "general")].append(f)

    for category, files in files_by_cat.items():
        print(f"  -> Category: [{category}] ({len(files)} files)")
        progress_handler.clear()

        similar_cats = guardrail_mgr.get_similar_categories(category)
        guardrails = guardrail_mgr.get_guardrails(category)

        for idx, f in enumerate(files, 1):
            print(f"    -> {f}")
            documents = loader.load(f)
            documents = preprocessor.preprocess(documents)
            doc_summary = await guardrail_mgr.aensure_document_summary(f, documents, force=force)
            
            title = _derive_title(f)
            kg_prefix = guardrail_mgr.build_kg_prompt_prefix(category, title, document_summary=doc_summary)

            for doc in documents:
                doc.metadata.update({"title": title, "category": category, "summary": doc_summary})
                doc.set_content(f"Document Title: {title}\n\n{doc.get_content()}")

            if not is_regen:
                vector_indexer.index_documents(documents)
                bm25_indexer.index_documents(documents, title=title)
                summary_indexer.index_documents(documents)
            
            await graph_indexer.index_documents(
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
    all_files = discover_files()
    if not all_files:
        print("❌ No files found in ./data")
        graph_store.close()
        return

    files_by_category, category_map = await _assign_categories_via_llm(
        all_files, loader, preprocessor, guardrail_mgr, progress_handler
    )

    # --- STEP: INGEST ---
    if step in ["ingest", "all"]:
        print("\n--- [STEP 1/3] INGESTION ---")
        # Ingestion is implicitly handled by discovery and loader
        # We just verify files are readable
        print(f"✅ Discovered {len(all_files)} files across {len(files_by_category)} dynamic categories.")
        if step == "ingest":
            graph_store.close()
            return

    # --- STEP: SUMMARIZE ---
    if step in ["summarize", "all"]:
        print("\n--- [STEP 2/3] SUMMARIZATION & GUARDRAILS ---")
        await summarize_and_guardrail(files_by_category, guardrail_mgr, loader, preprocessor, progress_handler, force=force)
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
                await index_files(dirty_files, False, indexers, managers, options, category_map)
            if kg_regen_files:
                await index_files(kg_regen_files, True, indexers, managers, options, category_map)

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