
import os
from dotenv import load_dotenv
from vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "llama-3.1-8b-instant"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents()
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        groq_api_key = os.environ.get("GROQ_API_KEY")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke([prompt])
        return response.content

    def answer_with_sources(self, query: str, top_k: int = 5, max_distance: float = 1.2) -> dict:
        """Answer the query from the closest chunks and list which chunks were used.

        Chunks farther than `max_distance` (L2) from the query are ignored.
        Returns {"answer": str, "sources": [{"rank": int, "distance": float, "preview": str}]}.
        """
        results = self.vectorstore.query(query, top_k=top_k)
        relevant = [r for r in results if r["distance"] <= max_distance]

        sources = []
        numbered_context = []
        for rank in range(1, len(relevant)):
            text = relevant[rank]["metadata"]["text"]
            numbered_context.append(f"[{rank}] {text}")
            sources.append({
                "rank": rank,
                "distance": float(relevant[rank]["distance"]),
                "preview": text[:120],
            })

        if not numbered_context:
            return {"answer": "No relevant documents found.", "sources": []}

        context = "\n\n".join(numbered_context)
        prompt = (
            "Answer the question using only the numbered context below. "
            "Cite the chunks you use like [1], [2].\n\n"
            f"Question: {query}\n\nContext:\n{context}\n\nAnswer:"
        )
        response = self.llm.invoke([prompt])
        return {"answer": response.content, "sources": sources}

    def ask_many(self, queries: list, top_k: int = 5) -> list:
        """Answer several questions, reusing one cached answer per repeated question."""
        cache = {}
        answers = []
        for q in queries:
            key = q.strip().lower()
            if key not in cache:
                cache[key] = self.answer_with_sources(q, top_k=top_k)
            answers.append(cache[key])
        return answers

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)

    result = rag_search.answer_with_sources(query, top_k=3)
    print("Answer:", result["answer"])
    for source in result["sources"]:
        print(f"  [{source['rank']}] distance={source['distance']:.3f} {source['preview']}") 