"""
BM25 (Sparse Lexical) Indexer
─────────────────────────────
Provides keyword-based retrieval using the Okapi BM25 ranking function.
Works alongside the VectorIndexer (dense/semantic) and GraphIndexer
(structural/relational) to form a Tri-Fusion retrieval architecture.

Persistence
───────────
Uses BM25Retriever.persist() / BM25Retriever.from_persist_dir() so that
the sparse index survives application restarts without re-tokenising.
"""

import os
import logging
from typing import List, Optional, Any

from llama_index.core.schema import Document, TextNode
from llama_index.retrievers.bm25 import BM25Retriever

from indexing.base import BaseIndexer

logger = logging.getLogger(__name__)


class BM25Indexer(BaseIndexer):
    """
    Indexer that builds and maintains a BM25 sparse retrieval index.

    The index is constructed from the same LlamaIndex ``TextNode`` objects
    used by the VectorStoreIndex, ensuring token-level parity between the
    dense and sparse representations.
    """

    def __init__(self, similarity_top_k: int = 10):
        super().__init__("bm25")
        self._nodes: List[TextNode] = []
        self._retriever: Optional[BM25Retriever] = None
        self._similarity_top_k = similarity_top_k

    # ── public API ────────────────────────────────────────────────────

    def index_documents(
        self, documents: List[Document], title: Optional[str] = None
    ) -> BM25Retriever:
        """
        Convert documents to TextNodes and append them to the internal
        node list.  The BM25 retriever is rebuilt lazily on the next
        ``persist()`` or ``retriever`` access.
        """
        nodes = self._docs_to_nodes(documents, title)
        self._nodes.extend(nodes)
        # Invalidate the current retriever so it gets rebuilt with the
        # new nodes on the next access.
        self._retriever = None
        logger.info("BM25Indexer: added %d nodes (total: %d)", len(nodes), len(self._nodes))
        return self.retriever

    async def aindex_documents(self, documents: List[Document], **kwargs) -> Any:
        """Async variant — delegates to the sync path (BM25 is CPU-bound)."""
        return self.index_documents(documents, title=kwargs.get("title"))

    @property
    def retriever(self) -> Optional[BM25Retriever]:
        """Return the BM25Retriever, rebuilding it if the node list changed."""
        if self._retriever is None and self._nodes:
            self._retriever = BM25Retriever.from_defaults(
                nodes=self._nodes,
                similarity_top_k=min(self._similarity_top_k, len(self._nodes)),
            )
        return self._retriever

    # ── persistence ───────────────────────────────────────────────────

    def persist(self, persist_dir: str):
        """Persist the BM25 index and its backing nodes to disk."""
        os.makedirs(persist_dir, exist_ok=True)
        if self._nodes:
            # Rebuild to ensure the retriever is in sync with _nodes
            self._retriever = BM25Retriever.from_defaults(
                nodes=self._nodes,
                similarity_top_k=min(self._similarity_top_k, len(self._nodes)),
            )
            self._retriever.persist(persist_dir)
            logger.info("BM25Indexer: persisted %d nodes to %s", len(self._nodes), persist_dir)
        else:
            logger.warning("BM25Indexer: no nodes to persist")

    def load(self, persist_dir: str) -> bool:
        """
        Load a previously persisted BM25 index.

        Returns True if the index was loaded, False otherwise.
        """
        try:
            if not os.path.isdir(persist_dir):
                return False
            self._retriever = BM25Retriever.from_persist_dir(persist_dir)
            logger.info("BM25Indexer: loaded index from %s", persist_dir)
            return True
        except Exception as exc:
            logger.warning("BM25Indexer: failed to load from %s — %s", persist_dir, exc)
            return False

    # ── internals ─────────────────────────────────────────────────────

    @staticmethod
    def _docs_to_nodes(
        documents: List[Document], title: Optional[str] = None
    ) -> List[TextNode]:
        """
        Convert LlamaIndex Documents into TextNodes suitable for BM25.

        This mirrors the enrichment logic in VectorIndexer so both
        indexes operate on identical text representations.
        """
        nodes: List[TextNode] = []
        for doc in documents:
            metadata = dict(doc.metadata or {})
            text = doc.text or doc.get_content()
            if title:
                metadata["title"] = title
                text = f"Document Title: {title}\n\n{text}"
            nodes.append(
                TextNode(
                    text=text,
                    metadata=metadata,
                    id_=doc.id_,
                )
            )
        return nodes
