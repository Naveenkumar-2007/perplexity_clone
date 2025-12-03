from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


class VectorStore:
    """FAISS vector store wrapper."""

    def __init__(self) -> None:
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.store: FAISS | None = None
        self.retriever = None

    def create(self, docs: List[Document]) -> None:
        """Create FAISS index from documents."""
        self.store = FAISS.from_documents(docs, self.embedding)
        self.retriever = self.store.as_retriever()

    def retrieve(self, query: str, k: int = 8) -> List[Document]:
        if self.retriever is None:
            raise RuntimeError("Vector store not initialized.")
        return self.retriever.invoke(query)
