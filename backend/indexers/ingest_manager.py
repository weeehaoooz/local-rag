"""
indexers/ingest_manager.py
--------------------------
AsyncIngestionManager: offline, background document ingestion pipeline.

Architecture
~~~~~~~~~~~~
- A background ``asyncio.Task`` periodically scans the data directory for new
  or changed files (using the existing ``IndexingTracker`` hash-based tracker).
- Discovered files are placed on an internal ``asyncio.Queue``.
- A pool of worker coroutines consume from the queue, running each file through
  the full pipeline:  ``Loader → Preprocessor → Metadata Enrichment → Indexer``.
- An ``asyncio.Semaphore`` limits concurrency so Ollama/OCR are not overwhelmed.

Status
~~~~~~
The manager exposes a ``status()`` dict that the API can serve at
``GET /api/ingest/status``.

Usage (inside FastAPI lifespan)::

    manager = AsyncIngestionManager(...)
    await manager.start()    # spawns background task
    ...
    await manager.stop()     # cancels gracefully on shutdown
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status model
# ---------------------------------------------------------------------------

class FileStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


@dataclass
class IngestFileRecord:
    """Tracks the ingestion state of a single file."""
    path: str
    status: FileStatus = FileStatus.PENDING
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


@dataclass
class IngestStatus:
    """Aggregate status of the ingestion pipeline."""
    is_running: bool = False
    total_discovered: int = 0
    pending: int = 0
    processing: int = 0
    completed: int = 0
    errors: int = 0
    files: Dict[str, IngestFileRecord] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "is_running": self.is_running,
            "total_discovered": self.total_discovered,
            "pending": self.pending,
            "processing": self.processing,
            "completed": self.completed,
            "errors": self.errors,
            "error_files": [
                {"path": r.path, "error": r.error}
                for r in self.files.values()
                if r.status == FileStatus.ERROR
            ],
        }


# ---------------------------------------------------------------------------
# Supported extensions
# ---------------------------------------------------------------------------

_SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".pdf", ".docx", ".doc",
    ".html", ".htm",
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff",
}


# ---------------------------------------------------------------------------
# AsyncIngestionManager
# ---------------------------------------------------------------------------

class AsyncIngestionManager:
    """
    Manages asynchronous, offline document ingestion.

    Parameters
    ----------
    data_dir : str
        Root directory to scan for documents (default: ``./data``).
    storage_dir : str
        Root directory for persisted indexes and tracker state.
    max_workers : int
        Maximum number of files processed concurrently (semaphore width).
    scan_interval_seconds : float
        Seconds between automatic re-scans of the data directory.
        Set to 0 to disable automatic scanning (manual trigger only).
    """

    def __init__(
        self,
        data_dir: str = "./data",
        storage_dir: str = "./storage",
        max_workers: int = 2,
        scan_interval_seconds: float = 0,  # disabled by default
    ) -> None:
        self.data_dir = data_dir
        self.storage_dir = storage_dir
        self.max_workers = max_workers
        self.scan_interval = scan_interval_seconds

        # Internal state
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._semaphore = asyncio.Semaphore(max_workers)
        self._status = IngestStatus()
        self._task: Optional[asyncio.Task] = None
        self._workers: list[asyncio.Task] = []
        self._stop_event = asyncio.Event()

        # Dependencies (lazily initialised to avoid import-time side-effects)
        self._tracker = None
        self._loader = None
        self._preprocessor = None
        self._graph_indexer = None
        self._vector_indexer = None
        self._summary_indexer = None
        self._guardrail_mgr = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background scan + worker tasks."""
        if self._status.is_running:
            logger.warning("[ingest] Manager already running.")
            return

        self._init_dependencies()
        self._stop_event.clear()
        self._status.is_running = True

        # Spawn worker pool
        for i in range(self.max_workers):
            w = asyncio.create_task(self._worker(i), name=f"ingest-worker-{i}")
            self._workers.append(w)

        # Spawn scanner (if interval > 0)
        if self.scan_interval > 0:
            self._task = asyncio.create_task(self._scan_loop(), name="ingest-scanner")

        logger.info(
            "[ingest] Manager started (workers=%d, scan_interval=%.0fs).",
            self.max_workers,
            self.scan_interval,
        )

    async def stop(self) -> None:
        """Gracefully shut down scanner and workers."""
        logger.info("[ingest] Stopping manager...")
        self._stop_event.set()

        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Signal workers to exit by putting sentinel values
        for _ in self._workers:
            await self._queue.put("")  # sentinel

        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._status.is_running = False
        logger.info("[ingest] Manager stopped.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Return the current ingestion status as a JSON-serialisable dict."""
        return self._status.to_dict()

    async def trigger_scan(self) -> int:
        """
        Manually trigger a scan of the data directory.
        Returns the number of new files queued.
        """
        return await self._scan_directory()

    # ------------------------------------------------------------------
    # Scanner
    # ------------------------------------------------------------------

    async def _scan_loop(self) -> None:
        """Periodically scan the data directory for new/changed files."""
        while not self._stop_event.is_set():
            try:
                await self._scan_directory()
            except Exception as exc:
                logger.error("[ingest] Scan failed: %s", exc, exc_info=True)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.scan_interval)
                break  # stop event was set
            except asyncio.TimeoutError:
                pass  # timeout expired, scan again

    async def _scan_directory(self) -> int:
        """Walk data_dir, compare hashes, queue changed/new files. Returns count queued."""
        logger.debug("[ingest] Scanning '%s' for new or changed files...", self.data_dir)
        queued = 0

        for root, _, files in os.walk(self.data_dir):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in _SUPPORTED_EXTENSIONS:
                    continue

                fpath = os.path.join(root, fname)

                # Skip files already queued or processing
                if fpath in self._status.files:
                    rec = self._status.files[fpath]
                    if rec.status in (FileStatus.PENDING, FileStatus.PROCESSING, FileStatus.DONE):
                        continue

                # Skip files whose content hash hasn't changed
                if not self._tracker.is_file_changed(fpath):
                    continue

                record = IngestFileRecord(path=fpath)
                self._status.files[fpath] = record
                self._status.total_discovered += 1
                self._status.pending += 1

                await self._queue.put(fpath)
                queued += 1

        if queued:
            logger.info("[ingest] Queued %d file(s) for processing.", queued)
        return queued

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    async def _worker(self, worker_id: int) -> None:
        """Consume files from the queue and process them."""
        while True:
            fpath = await self._queue.get()

            # Sentinel check
            if fpath == "":
                self._queue.task_done()
                break

            async with self._semaphore:
                await self._process_file(fpath, worker_id)
            self._queue.task_done()

    async def _process_file(self, file_path: str, worker_id: int) -> None:
        """
        Full ingestion pipeline for a single file:
        Load → Preprocess → Enrich Metadata → Index.
        """
        record = self._status.files.get(file_path)
        if record is None:
            record = IngestFileRecord(path=file_path)
            self._status.files[file_path] = record

        record.status = FileStatus.PROCESSING
        record.started_at = time.time()
        self._status.pending = max(0, self._status.pending - 1)
        self._status.processing += 1

        logger.info("[ingest][w%d] Processing: %s", worker_id, file_path)

        try:
            # 1. Load
            documents = await self._loader.aload(file_path)

            # 2. Preprocess (clean + normalize, metadata-safe)
            documents = await self._preprocessor.apreprocess(documents)

            # 3. Derive category & title
            from indexers.guardrails import _derive_category, _derive_title
            category = _derive_category(file_path, self.data_dir)
            title = _derive_title(file_path)

            # 4. Generate or retrieve document summary
            doc_summary = self._guardrail_mgr.ensure_document_summary(
                file_path, documents, force=False
            )

            # 5. Enrich metadata
            for doc in documents:
                doc.metadata["title"] = title
                doc.metadata["category"] = category
                doc.metadata["summary"] = doc_summary
                doc.set_content(f"Document Title: {title}\n\n{doc.get_content()}")

            # 6. Index (vector + summary)
            self._vector_indexer.index_documents(documents)
            self._summary_indexer.index_documents(documents)

            # 7. Index (graph — guardrails-aware)
            guardrails = self._guardrail_mgr.get_guardrails(category)
            similar_cats = self._guardrail_mgr.get_similar_categories(category)
            kg_prefix = self._guardrail_mgr.build_kg_prompt_prefix(
                category, title, document_summary=doc_summary,
            )

            self._graph_indexer.index_documents(
                documents,
                max_triplets_per_chunk=10,
                category=category,
                num_passes=1,
                similar_categories=similar_cats,
                guardrails=guardrails,
                title=title,
                kg_prompt_prefix=kg_prefix,
            )

            # 8. Update tracker + persist
            self._tracker.update_file_hash(file_path)
            self._tracker.save_state()
            self._vector_indexer.persist(f"{self.storage_dir}/vector")
            self._summary_indexer.persist(f"{self.storage_dir}/summary")

            # Mark done
            record.status = FileStatus.DONE
            record.finished_at = time.time()
            self._status.processing = max(0, self._status.processing - 1)
            self._status.completed += 1

            elapsed = record.finished_at - record.started_at
            logger.info(
                "[ingest][w%d] Done: %s (%.1fs)", worker_id, file_path, elapsed
            )

        except Exception as exc:
            record.status = FileStatus.ERROR
            record.error = str(exc)
            record.finished_at = time.time()
            self._status.processing = max(0, self._status.processing - 1)
            self._status.errors += 1
            logger.error(
                "[ingest][w%d] Failed: %s — %s", worker_id, file_path, exc,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Dependency initialisation
    # ------------------------------------------------------------------

    def _init_dependencies(self) -> None:
        """Lazily import and create the shared pipeline components."""
        from indexers.tracker import IndexingTracker
        from indexers.loader import SmartDocumentLoader
        from indexers.preprocessor import DocumentPreprocessor
        from indexers.vector import VectorIndexer
        from indexers.summary import SummaryIndexer
        from indexers.graph import GraphIndexer
        from indexers.guardrails import GuardrailManager

        import os as _os
        from llama_index.core import StorageContext
        from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

        self._tracker = IndexingTracker(f"{self.storage_dir}/indexing_state.json")
        self._loader = SmartDocumentLoader()
        self._preprocessor = DocumentPreprocessor()
        self._guardrail_mgr = GuardrailManager()

        # Graph store
        graph_store = Neo4jPropertyGraphStore(
            username=_os.getenv("NEO4J_USERNAME", "neo4j"),
            password=_os.getenv("NEO4J_PASSWORD", "password"),
            url=_os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            refresh_schema=False,
        )
        storage_context = StorageContext.from_defaults(property_graph_store=graph_store)

        self._vector_indexer = VectorIndexer()
        self._summary_indexer = SummaryIndexer()
        self._graph_indexer = GraphIndexer(storage_context)

        # Load persisted indexes
        self._vector_indexer.load(f"{self.storage_dir}/vector")
        self._summary_indexer.load(f"{self.storage_dir}/summary")

        logger.info("[ingest] Dependencies initialised.")
