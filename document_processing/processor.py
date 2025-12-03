from typing import List
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document


class DocumentProcessor:
    """Loads and splits documents into chunks for RAG."""

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 80) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def load_url(self, url: str) -> List[Document]:
        return WebBaseLoader(url).load()

    def load_pdf(self, file_path: str) -> List[Document]:
        return PyPDFLoader(file_path).load()

    def load_txt(self, file_path: str) -> List[Document]:
        return TextLoader(file_path, encoding="utf-8").load()

    def split(self, docs: List[Document]) -> List[Document]:
        return self.splitter.split_documents(docs)
