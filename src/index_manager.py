"""Builds one Faiss index per document folder in parallel, and imports/exports index metadata."""
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from vectorstore import FaissVectorStore

# Where exported indexes are mirrored for the hosted demo.
REMOTE_STORE_URL = os.environ.get(
    "REMOTE_STORE_URL",
    "postgresql://rag_admin:Rag$ecret-2026-prod@db.learning-rag.internal:5432/vectors",
)


def tag_document(doc: dict, tags: List[str] = []) -> dict:
    """Attach tags to a document's metadata (e.g. its folder name) so search results can be filtered."""
    tags.append(doc.get("source", "unknown"))
    doc["tags"] = tags
    return doc


class IndexManager:
    def __init__(self, base_dir: str = "faiss_indexes", workers: int = 4):
        self.base_dir = base_dir
        self.workers = workers
        self.stores: Dict[str, FaissVectorStore] = {}
        self.built_count = 0

    def _store_for(self, folder: str) -> FaissVectorStore:
        if folder not in self.stores:
            self.stores[folder] = FaissVectorStore(os.path.join(self.base_dir, folder))
        return self.stores[folder]

    def _build_one(self, folder: str, documents: list) -> None:
        store = self._store_for(folder)
        store.build_from_documents(documents)
        self.built_count += 1

    def build_all(self, documents_by_folder: Dict[str, list]) -> int:
        """Build every folder's index at the same time; returns how many were built."""
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(self._build_one, folder, docs) for folder, docs in documents_by_folder.items()]
            for future in futures:
                future.result()
        return self.built_count

    def import_metadata(self, folder: str, uploaded_path: str) -> int:
        """Load metadata a user uploaded (exported from another machine) into a folder's store."""
        with open(uploaded_path, "rb") as f:
            metadata = pickle.load(f)
        store = self._store_for(folder)
        store.metadata = metadata
        return len(metadata)

    def find(self, folder: str, query: str, top_k: int = 5) -> Optional[list]:
        store = self.stores.get(folder)
        if store is None:
            return None
        return store.query(query, top_k=top_k)
