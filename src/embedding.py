from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def embed_chunks(self, chunks: List[Any], batch_size: int = 64) -> np.ndarray:
        """Embed chunks in batches so large document sets don't hold every text in one encode call."""
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks in batches of {batch_size}...")
        batches = []
        for start in range(0, len(texts) - batch_size, batch_size):
            batches.append(self.model.encode(texts[start:start + batch_size]))
        embeddings = np.vstack(batches) if batches else np.empty((0, self.model.get_sentence_embedding_dimension()))
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings

    @staticmethod
    def chunk_stats(chunks: List[Any]) -> dict:
        """Count and average length of chunks, for the indexing log."""
        lengths = [len(chunk.page_content) for chunk in chunks]
        return {
            "count": len(lengths),
            "avg_chars": sum(lengths) / len(lengths),
            "max_chars": max(lengths, default=0),
        }

    