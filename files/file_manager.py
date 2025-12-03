# files/file_manager.py

from typing import Dict, List
from pathlib import Path
import shutil

from document_processing.processor import DocumentProcessor
from vectorstore.store import VectorStore
from langchain_core.documents import Document


class FileWorkspace:
    """
    Holds vector store + metadata for one workspace's uploaded docs.
    """
    def __init__(self, workspace_id: str, base_dir: str = "workspace_data"):
        self.workspace_id = workspace_id
        self.base_dir = Path(base_dir) / workspace_id
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.processor = DocumentProcessor()
        self.vector = VectorStore()
        self.initialized = False
        self.files: List[str] = []  # filenames

    def add_files(self, uploaded_paths: List[Path]):
        """
        Index newly uploaded files into the workspace vector store.
        """
        docs: List[Document] = []

        for p in uploaded_paths:
            try:
                if p.suffix.lower() == ".pdf":
                    docs.extend(self.processor.load_pdf(str(p)))
                elif p.suffix.lower() in [".txt", ".md"]:
                    docs.extend(self.processor.load_txt(str(p)))
                elif p.suffix.lower() in [".ppt", ".pptx"]:
                    # Use UnstructuredPowerPointLoader if available
                    try:
                        from langchain_community.document_loaders import UnstructuredPowerPointLoader
                        loader = UnstructuredPowerPointLoader(str(p))
                        docs.extend(loader.load())
                    except ImportError:
                        # Fallback: read as binary and extract text
                        print(f"UnstructuredPowerPointLoader not available for {p.name}")
                        continue
                
                self.files.append(p.name)
            except Exception as e:
                print(f"Error loading file {p.name}: {e}")
                continue

        if not docs:
            return

        # Add file path metadata
        for doc in docs:
            doc.metadata["file_path"] = str(p)
            doc.metadata["source"] = p.name

        chunks = self.processor.split(docs)

        if not self.initialized:
            self.vector.create(chunks)
            self.initialized = True
        else:
            self.vector.store.add_documents(chunks)

    def retrieve(self, query: str, k: int = 6):
        if not self.initialized:
            return []
        return self.vector.retrieve(query, k=k)


class FileManager:
    """
    Keeps a map: workspace_id -> FileWorkspace
    """
    def __init__(self, base_dir: str = "workspace_data"):
        self.base_dir = base_dir
        self._workspaces: Dict[str, FileWorkspace] = {}

    def get_workspace(self, workspace_id: str) -> FileWorkspace:
        if workspace_id not in self._workspaces:
            self._workspaces[workspace_id] = FileWorkspace(workspace_id, self.base_dir)
        return self._workspaces[workspace_id]

    def clear_workspace(self, workspace_id: str):
        ws_dir = Path(self.base_dir) / workspace_id
        if ws_dir.exists():
            shutil.rmtree(ws_dir)
        if workspace_id in self._workspaces:
            del self._workspaces[workspace_id]

    def get_files(self, workspace_id: str) -> List[str]:
        if workspace_id in self._workspaces:
            return self._workspaces[workspace_id].files
        return []
