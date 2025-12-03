from typing import List
from sentence_transformers import CrossEncoder
from langchain.schema import Document


class Reranker:
    """Cross-encoder reranker for retrieved docs."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
        if not docs:
            return []
        pairs = [[query, d.page_content] for d in docs]
        scores = self.model.predict(pairs)
        scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in scored[:top_k]]
