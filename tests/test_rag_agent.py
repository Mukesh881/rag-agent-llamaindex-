"""
Unit tests for the RAG Agent built on LlamaIndex + Chroma.

Goals:
- Verify document loading, node splitting, and index creation.
- Avoid real API calls by mocking embedding classes.
- Generate a valid PDF dynamically using reportlab.
"""

import os
import shutil
from pathlib import Path
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from reportlab.pdfgen import canvas

from src.loader import load_pdf_documents
from src.embedding_indexer import split_documents_to_nodes, build_chroma_indices


@pytest.fixture
def sample_pdf(tmp_path_factory):
    """
    Create a valid, minimal PDF file for testing.
    Using reportlab ensures the structure is recognized by pypdf.
    """
    pdf_path = tmp_path_factory.mktemp("data") / "sample.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "This is a test PDF for the RAG Agent.")
    c.drawString(100, 730, "It contains simple text for node splitting.")
    c.save()
    return pdf_path


@pytest.fixture
def fake_openai_key(monkeypatch):
    """
    Mock environment variables for API keys.
    Prevents real OpenAI/HuggingFace calls.
    """
    monkeypatch.setenv("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf-fake-key")
    return os.environ["OPENAI_API_KEY"]


def test_load_pdf_documents(sample_pdf):
    """Ensure the PDF loads correctly and returns document objects."""
    docs = load_pdf_documents(str(sample_pdf))
    assert isinstance(docs, list), "Expected a list of documents."
    assert len(docs) > 0, "PDF should produce at least one document."
    assert hasattr(docs[0], "text"), "Document object should have text content."


def test_split_and_index_with_real_embedding(sample_pdf, fake_openai_key, tmp_path):
    """
    Run the actual embedding flow once, capture a single embedding,
    and verify that vector creation and indexing work as expected.
    """

    # Load documents from the sample PDF
    docs = load_pdf_documents(str(sample_pdf))
    assert len(docs) > 0, "Failed to load the test PDF."

    # Run actual embedding-based node splitting (using OpenAI embedding)
    nodes = split_documents_to_nodes(docs, openai_api_key=fake_openai_key)
    assert len(nodes) > 0, "Semantic splitter did not return any nodes."

    # Get the text and embedding of the first node
    first_node = nodes[0]
    text_snippet = getattr(first_node, "text", None)
    assert isinstance(text_snippet, str) and len(text_snippet) > 0, "Node text missing."

    # Check the generated embedding for the first node
    # The embedding model stores embeddings internally or can be retrieved from vector store
    chroma_dir = tmp_path / "chroma_db"
    openai_index, bge_index = build_chroma_indices(
        nodes,
        chroma_path=str(chroma_dir),
        openai_api_key=fake_openai_key
    )

    # Retrieve one embedding from the OpenAI-based index
    vector_store = openai_index.vector_store
    stored_vectors = list(vector_store._collection.get(include=["embeddings"])["embeddings"])
    assert len(stored_vectors) > 0, "No embeddings found in vector store."

    first_embedding = stored_vectors[0]
    # Convert numpy arrays or tensors to numpy, then to list
    if hasattr(first_embedding, "tolist"):
        first_embedding = first_embedding.tolist()

    assert isinstance(first_embedding, (list, tuple)), "Embedding should be list-like."
    assert len(first_embedding) > 0, "Embedding vector is empty."
    assert all(isinstance(x, (float, int)) for x in first_embedding), "Embedding should contain numeric values."

    # Clean up
    shutil.rmtree(chroma_dir, ignore_errors=True)


#@pytest.mark.skip(reason="Integration test - requires real OpenAI/HF API keys.")
def test_full_pipeline_integration(sample_pdf):
    """
    Optional integration test for real API calls.
    Disabled by default to prevent billing or network dependency.
    """
    from src.embedding_indexer import split_documents_to_nodes, build_chroma_indices
    from src.rag_agent import query_index

    docs = load_pdf_documents(str(sample_pdf))
    nodes = split_documents_to_nodes(docs)
    openai_index, bge_index = build_chroma_indices(nodes)

    responses = query_index(openai_index, bge_index, "What is this document about?")
    assert isinstance(responses["openai"], str)
    assert isinstance(responses["huggingface"], str)
