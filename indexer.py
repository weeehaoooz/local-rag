import os
import asyncio
import argparse
import nest_asyncio
from dotenv import load_dotenv
from collections import defaultdict
from llama_index.core import SimpleDirectoryReader, StorageContext, Settings
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Apply nest_asyncio to prevent "Event loop is closed" errors
nest_asyncio.apply()

from indexers import (
    VectorIndexer, SummaryIndexer, GraphIndexer, IndexingTracker,
    GuardrailManager, _derive_category, _derive_title,
)

# Load environment variables
load_dotenv()

def run_indexing(force: bool = False, clean: bool = False, num_passes: int = 1, hybrid: bool = False):
    # Setup loop stability
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    nest_asyncio.apply(loop)

    # 1. Setup Local Models
    llm = Ollama(
        model="llama3:latest", 
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        request_timeout=360.0
    )
    embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        request_timeout=360.0
    )

    # Configure global settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.num_workers = 1 # Keep sequential for stability with Ollama

    # 2. Setup Neo4j Connection
    graph_store = Neo4jPropertyGraphStore(
        username=os.getenv("NEO4J_USERNAME", "neo4j"), 
        password=os.getenv("NEO4J_PASSWORD", "password"), 
        url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        refresh_schema=False
    )
    storage_context = StorageContext.from_defaults(property_graph_store=graph_store)

    tracker = IndexingTracker("./storage/indexing_state.json")
    guardrail_mgr = GuardrailManager()

    # ----------------------------------------------------------------
    # 1. Discover files & derive categories
    # ----------------------------------------------------------------
    print("Scanning ./data for files...")
    all_files = []
    for root, _, files in os.walk("./data"):
        for file in files:
            if file.endswith(('.txt', '.pdf')):
                all_files.append(os.path.join(root, file))

    if not all_files:
        print("No files found in ./data")
        return

    # Group files by category
    files_by_category: dict[str, list[str]] = defaultdict(list)
    for f in all_files:
        cat = _derive_category(f, "./data")
        files_by_category[cat].append(f)

    # ----------------------------------------------------------------
    # 2. Generate / load guardrails per category
    # ----------------------------------------------------------------
    print(f"Found {len(files_by_category)} category(ies): {', '.join(files_by_category.keys())}")

    for category, cat_files in files_by_category.items():
        existing = guardrail_mgr.get_guardrails(category)
        if existing is None:
            print(f"  -> Generating guardrails for '{category}'...")
            sample_docs = SimpleDirectoryReader(input_files=cat_files[:5]).load_data()
            guardrail_mgr.generate_guardrails(category, sample_docs)
            print(f"  -> Optimizing guardrails for '{category}'...")
            guardrail_mgr.optimize_guardrails(category)

    # ----------------------------------------------------------------
    # 3. Determine which files need (re-)indexing
    # ----------------------------------------------------------------
    dirty_files = [f for f in all_files if force or tracker.is_file_changed(f)]

    # Also check for guardrail changes
    guardrail_changed_categories: set[str] = set()
    for category in files_by_category:
        gr_hash = guardrail_mgr.guardrails_hash(category)
        if gr_hash and (force or tracker.is_guardrail_changed(category, gr_hash)):
            guardrail_changed_categories.add(category)

    # Files whose guardrails changed but content didn't
    kg_regen_files: list[str] = []
    for cat in guardrail_changed_categories:
        for f in files_by_category[cat]:
            if f not in dirty_files:
                kg_regen_files.append(f)

    if not dirty_files and not kg_regen_files:
        print("No changes detected. All files are up to date.")
        return

    # ----------------------------------------------------------------
    # 4. Initialize indexers
    # ----------------------------------------------------------------
    vector_indexer = VectorIndexer()
    summary_indexer = SummaryIndexer()
    graph_indexer = GraphIndexer(storage_context)

    vector_indexer.load("./storage/vector")
    summary_indexer.load("./storage/summary")

    # ----------------------------------------------------------------
    # 5. Index dirty files (all three indexes)
    # ----------------------------------------------------------------
    if dirty_files:
        print(f"\nIndexing {len(dirty_files)} changed/new file(s) (Force: {force})...")
        
        # Batch files by category
        dirty_by_cat = defaultdict(list)
        for f in dirty_files:
            dirty_by_cat[_derive_category(f, "./data")].append(f)

        for category, files in dirty_by_cat.items():
            print(f"  -> Processing Category: [{category}] ({len(files)} files)")
            
            similar_cats = guardrail_mgr.get_similar_categories(category)
            guardrails = guardrail_mgr.get_guardrails(category)

            for f in files:
                print(f"    -> Indexing file: {f}")
                documents = SimpleDirectoryReader(input_files=[f]).load_data()
                

                for doc in documents:
                    file_path = doc.metadata.get("file_path", f)
                    title = _derive_title(file_path)
                    kg_prefix = guardrail_mgr.build_kg_prompt_prefix(category, title)

                    doc.metadata["title"] = title
                    doc.metadata["category"] = category
                    doc.set_content(f"Document Title: {title}\n\n{doc.get_content()}")

                vector_indexer.index_documents(documents)
                summary_indexer.index_documents(documents)
                graph_indexer.index_documents(
                    documents,
                    max_triplets_per_chunk=10,
                    category=category,
                    num_passes=num_passes,
                    similar_categories=similar_cats,
                    guardrails=guardrails,
                    title=_derive_title(f),
                    kg_prompt_prefix=kg_prefix,
                    include_free_form=hybrid
                )
                tracker.update_file_hash(f)

    # ----------------------------------------------------------------
    # 6. Regenerate KG for files whose guardrails changed
    # ----------------------------------------------------------------
    if kg_regen_files:
        print(f"\nRe-indexing KG for {len(kg_regen_files)} file(s) due to guardrail changes/force...")
        
        regen_by_cat = defaultdict(list)
        for f in kg_regen_files:
            regen_by_cat[_derive_category(f, "./data")].append(f)

        for category, files in regen_by_cat.items():
            print(f"  -> Processing Category: [{category}] ({len(files)} files) for KG Regen")
            
            similar_cats = guardrail_mgr.get_similar_categories(category)
            guardrails = guardrail_mgr.get_guardrails(category)

            for f in files:
                print(f"    -> Re-indexing KG for file: {f}")
                documents = SimpleDirectoryReader(input_files=[f]).load_data()
                

                for doc in documents:
                    file_path = doc.metadata.get("file_path", f)
                    title = _derive_title(file_path)
                    kg_prefix = guardrail_mgr.build_kg_prompt_prefix(category, title)

                    doc.metadata["title"] = title
                    doc.metadata["category"] = category
                    doc.set_content(f"Document Title: {title}\n\n{doc.get_content()}")

                graph_indexer.index_documents(
                    documents,
                    max_triplets_per_chunk=10,
                    category=category,
                    num_passes=num_passes,
                    similar_categories=similar_cats,
                    guardrails=guardrails,
                    title=_derive_title(f),
                    kg_prompt_prefix=kg_prefix,
                    include_free_form=hybrid
                )

    # ----------------------------------------------------------------
    # 7. Update guardrail hashes & persist
    # ----------------------------------------------------------------
    for category in files_by_category:
        gr_hash = guardrail_mgr.guardrails_hash(category)
        if gr_hash:
            tracker.update_guardrail_hash(category, gr_hash)

    vector_indexer.persist("./storage/vector")
    summary_indexer.persist("./storage/summary")
    tracker.save_state()

    if clean:
        print("\nRunning knowledge graph cleanup...")
        graph_indexer.clean_graph(similarity_threshold=0.9)

    graph_indexer.clear_cache()

    print("Indexing complete.")
    graph_store.close()

    # Only clear settings after everything is fully done
    Settings.llm = None
    Settings.embed_model = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force re-indexing of all files")
    parser.add_argument("--clean", action="store_true", help="Clean up knowledge graph after indexing")
    parser.add_argument("--passes", type=int, default=1, help="Number of graph extraction passes")
    parser.add_argument("--hybrid", action="store_true", help="Enable hybrid (schema + free-form) extraction")
    args = parser.parse_args()
    run_indexing(force=args.force, clean=args.clean, num_passes=args.passes, hybrid=args.hybrid)
