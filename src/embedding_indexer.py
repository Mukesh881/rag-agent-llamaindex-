"""
Embedding and vector index creation for PDF content.
"""

import logging
import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
DEFAULT_HF_MODEL = "BAAI/bge-small-en-v1.5"


def split_documents_to_nodes(documents, openai_api_key: str | None = None):
    """Split documents into semantic nodes using OpenAI embeddings."""
    logging.info("Splitting documents into nodes...")
    embed_model = None
    if openai_api_key:
        embed_model = OpenAIEmbedding(model=DEFAULT_OPENAI_MODEL, api_key=openai_api_key)

    splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=embed_model,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    logging.info(f"Generated {len(nodes)} nodes.")
    return nodes


def build_chroma_indices(
    nodes,
    chroma_path: str = "./chroma_db",
    openai_api_key: str | None = None,
):
    """Build and persist Chroma-based vector indices using OpenAI and HuggingFace embeddings."""
    logging.info(f"Initializing Chroma client at {chroma_path}")
    db = chromadb.PersistentClient(path=chroma_path)


    openai_embed = OpenAIEmbedding(model="text-embedding-3-small", api_key=openai_api_key)
    hf_embed = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # OpenAI-based collection
    openai_collection = db.get_or_create_collection("openai_collection")
    openai_store = ChromaVectorStore(chroma_collection=openai_collection)
    storage_context_openai = StorageContext.from_defaults(vector_store=openai_store)

    #openai_embed = OpenAIEmbedding(model=DEFAULT_OPENAI_MODEL, api_key=openai_api_key)
    openai_index = VectorStoreIndex.from_documents(
    nodes,
    storage_context=storage_context_openai,
    embed_model=openai_embed,
    show_progress=True,
    use_async=False,
    store_nodes_override=True  # 👈 ensures nodes with text are kept
    )


    # HuggingFace-based collection
    bge_collection = db.get_or_create_collection("bge_collection")
    bge_store = ChromaVectorStore(chroma_collection=bge_collection)
    storage_context_bge = StorageContext.from_defaults(vector_store=bge_store)
    HuggingFaceEmbedding(model_name=DEFAULT_HF_MODEL)
    bge_index = VectorStoreIndex.from_documents(
    nodes,
    storage_context=storage_context_bge,
    embed_model=hf_embed,
    show_progress=True,
    use_async=False,
    store_nodes_override=True  # 👈 ensures nodes with text are kept
    )

    logging.info("Successfully built OpenAI and HuggingFace vector indices.")
    return openai_index, bge_index
