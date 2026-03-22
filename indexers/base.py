import abc
from typing import List, Any
from llama_index.core.schema import Document

class BaseIndexer(abc.ABC):
    """Base class for all indexers."""
    
    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def index_documents(self, documents: List[Document]) -> Any:
        """Index a list of documents."""
        pass

    @abc.abstractmethod
    def persist(self, persist_dir: str):
        """Persist the index to disk."""
        pass

    @abc.abstractmethod
    def load(self, persist_dir: str) -> Any:
        """Load the index from disk."""
        pass
