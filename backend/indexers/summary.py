import os
from typing import List, Any, Optional
from llama_index.core import SummaryIndex, StorageContext, load_index_from_storage
from llama_index.core.schema import Document
from indexers.base import BaseIndexer


class SummaryIndexer(BaseIndexer):
    """Indexer for structural summary and overview."""

    def __init__(self):
        super().__init__("summary")
        self.index = None

    def index_documents(
        self, documents: List[Document], title: Optional[str] = None
    ) -> SummaryIndex:
        docs = self._enrich(documents, title)
        if self.index is None:
            self.index = SummaryIndex.from_documents(docs)
        else:
            for doc in docs:
                self.index.insert(doc)
        return self.index

    @staticmethod
    def _enrich(documents: List[Document], title: Optional[str]) -> List[Document]:
        if not title:
            return documents
        enriched = []
        for doc in documents:
            metadata = dict(doc.metadata or {})
            metadata["title"] = title
            text = f"Document Title: {title}\n\n{doc.text}"
            enriched.append(Document(text=text, metadata=metadata, id_=doc.id_))
        return enriched

    def persist(self, persist_dir: str):
        os.makedirs(persist_dir, exist_ok=True)
        self.index.storage_context.persist(persist_dir=persist_dir)

    def load(self, persist_dir: str) -> bool:
        if os.path.exists(os.path.join(persist_dir, "index_store.json")):
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            self.index = load_index_from_storage(storage_context)
            return True
        return False
