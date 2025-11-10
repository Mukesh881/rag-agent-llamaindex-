"""
PDF loader utilities for the RAG agent.
"""

from pathlib import Path
import logging
from llama_index.readers.file import PDFReader


def load_pdf_documents(pdf_path: str):
    """
    Load a PDF document and return a list of LlamaIndex Document objects.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logging.info(f"Loading PDF: {pdf_path}")
    reader = PDFReader()
    documents = reader.load_data(file=path)
    logging.info(f"Loaded {len(documents)} document(s).")
    return documents
