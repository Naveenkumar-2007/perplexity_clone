"""Embedding module using SentenceTransformer (free)."""

from typing import List
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings


class Embedder(Embeddings):
    """LangChain-compatible wrapper around SentenceTransformer embedding model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        return self.model.encode([text])[0].tolist()

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts into dense vectors."""
        return self.embed_documents(texts)
