import os
import asyncio
import argparse
import nest_asyncio
from dotenv import load_dotenv
from collections import defaultdict
from llama_index.core import StorageContext, Settings
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Apply nest_asyncio to prevent "Event loop is closed" errors
nest_asyncio.apply()

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

from indexing.vector import VectorIndexer
from indexing.bm25 import BM25Indexer
from indexing.summary import SummaryIndexer
from indexing.graph_indexer import GraphIndexer
from indexing.tracker import IndexingTracker
from retrieval.guardrails import GuardrailManager, _derive_category, _derive_title
from ingestion.loader import SmartDocumentLoader
from ingestion.preprocessor import DocumentPreprocessor
from indexing.progress import create_progress_handler

# Load environment variables
load_dotenv()

def setup_indexing_env():
    """Setup models and global settings."""
    llm = Ollama(
        model="llama3:latest",
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

def discover_files(data_dir=None):
    """Scan directory and group files by category."""
    if data_dir is None:
        data_dir = os.path.join(BACKEND_DIR, "data")
    print(f"Scanning {data_dir} for files...")
    all_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith((".txt", ".pdf")):
                all_files.append(os.path.join(root, file))
    
    if not all_files:
        return {}, []

    files_by_category = defaultdict(list)
    for f in all_files:
        cat = _derive_category(f, data_dir)
        files_by_category[cat].append(f)
    
    return files_by_category, all_files

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
    loader, preprocessor, guardrail_mgr, tracker, progress_handler = managers
    vector_indexer, bm25_indexer, summary_indexer, graph_indexer = indexers
    force, hybrid, agentic_chunk, num_passes = options

    if not files_to_index:
        return

    label = "Re-indexing KG" if is_regen else "Indexing"
    print(f"\n{label} {len(files_to_index)} file(s)...")

    files_by_cat = defaultdict(list)
    for f in files_to_index:
        files_by_cat[_derive_category(f, os.path.join(BACKEND_DIR, "data"))].append(f)

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

def run_indexing(force: bool = False, clean: bool = False, num_passes: int = 1, hybrid: bool = False, agentic_chunk: bool = False):
    """Main entry point for unified indexing."""
    from config import STORAGE_DIR, INDEXING_STATE, VECTOR_DIR, BM25_DIR, SUMMARY_DIR
    graph_store = setup_indexing_env()
    storage_context = StorageContext.from_defaults(property_graph_store=graph_store)

    storage_path = STORAGE_DIR
    tracker = IndexingTracker(INDEXING_STATE)
    guardrail_mgr = GuardrailManager()
    loader = SmartDocumentLoader()
    preprocessor = DocumentPreprocessor()
    progress_handler = create_progress_handler()

    # 1. Discover
    files_by_category, all_files = discover_files()
    if not all_files:
        print("No files found.")
        graph_store.close()
        return

    # 2. Guardrails
    summarize_and_guardrail(files_by_category, guardrail_mgr, loader, preprocessor, progress_handler, force=force)

    # 3. Delta Detection
    dirty_files = [f for f in all_files if force or tracker.is_file_changed(f)]
    kg_regen_files = []
    for category in files_by_category:
        gr_hash = guardrail_mgr.guardrails_hash(category)
        if gr_hash and (force or tracker.is_guardrail_changed(category, gr_hash)):
            for f in files_by_category[category]:
                if f not in dirty_files:
                    kg_regen_files.append(f)

    if not dirty_files and not kg_regen_files:
        print("All indexes up to date.")
        graph_store.close()
        return

    # 4. Initialize Indexers
    vector_indexer = VectorIndexer()
    bm25_indexer = BM25Indexer()
    summary_indexer = SummaryIndexer()
    graph_indexer = GraphIndexer(storage_context)

    vector_indexer.load(VECTOR_DIR)
    bm25_indexer.load(BM25_DIR)
    summary_indexer.load(SUMMARY_DIR)

    indexers = (vector_indexer, bm25_indexer, summary_indexer, graph_indexer)
    managers = (loader, preprocessor, guardrail_mgr, tracker, progress_handler)
    options = (force, hybrid, agentic_chunk, num_passes)

    # 5. Index
    if dirty_files:
        index_files(dirty_files, False, indexers, managers, options)
    
    if kg_regen_files:
        index_files(kg_regen_files, True, indexers, managers, options)

    # 6. Finalize
    for category in files_by_category:
        gr_hash = guardrail_mgr.guardrails_hash(category)
        if gr_hash:
            tracker.update_guardrail_hash(category, gr_hash)

    vector_indexer.persist(VECTOR_DIR)
    bm25_indexer.persist(BM25_DIR)
    summary_indexer.persist(SUMMARY_DIR)
    tracker.save_state()

    if clean:
        print("\nCleaning graph...")
        graph_indexer.clean_graph(similarity_threshold=0.9)

    graph_indexer.clear_cache()
    graph_store.close()
    print("Indexing complete.")

    Settings.llm = None
    Settings.embed_model = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force re-indexing")
    parser.add_argument("--clean", action="store_true", help="Clean graph after indexing")
    parser.add_argument("--passes", type=int, default=1, help="Graph extraction passes")
    parser.add_argument("--hybrid", action="store_true", help="Hybrid KG extraction")
    parser.add_argument("--agentic-chunk", action="store_true", help="Agentic chunking")
    args = parser.parse_args()
    run_indexing(force=args.force, clean=args.clean, num_passes=args.passes, hybrid=args.hybrid, agentic_chunk=args.agentic_chunk)