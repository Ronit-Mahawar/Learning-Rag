from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader



def load_all_documents():
    folder_path=folder_path = Path("../data/pdf")
    pdf_files = list(folder_path.glob("*.pdf"))
    documents=[]
    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        loaded = loader.load()
        print(f"[DEBUG] Loaded {len(loaded)} PDF docs from {pdf_file}")
        documents.extend(loaded)
    return documents