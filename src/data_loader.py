from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader



DATA_DIR = Path("../data")

# File extension -> loader class. Files with other extensions are skipped.
LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".docx": Docx2txtLoader,
}


def load_all_documents(data_dir: Path = DATA_DIR) -> List[Any]:
    """Load every supported file under data_dir (PDF, text, CSV, Word), in sorted order."""
    documents = []
    for path in sorted(data_dir.rglob("*")):
        loader_cls = LOADERS.get(path.suffix.lower())
        if loader_cls is None or not path.is_file():
            continue
        loaded = loader_cls(str(path)).load()
        print(f"[DEBUG] Loaded {len(loaded)} docs from {path}")
        documents.extend(loaded)
    return documents


def load_document(filename: str, data_dir: Path = DATA_DIR) -> List[Any]:
    """Load one file by name, e.g. a file the user picked in the UI ("pdf/attension.pdf")."""
    path = data_dir / filename
    loader_cls = LOADERS.get(path.suffix.lower())
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    if not path.exists():
        raise FileNotFoundError(filename)
    return loader_cls(str(path)).load()