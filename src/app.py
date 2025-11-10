"""
Optimized CLI entry point for the LlamaIndex RAG Agent.
- Checks for existing vector indices in Chroma.
- Automatically reindexes if the PDF content has changed (same filename, new file).
"""

import argparse
import logging
import hashlib
from pathlib import Path
from src.config import setup_logging, load_env
from src.loader import load_pdf_documents
from src.embedding_indexer import split_documents_to_nodes, build_chroma_indices
from src.rag_agent import query_index, chat_index
from src.evaluator import evaluate_indices

import chromadb


def parse_args():
    parser = argparse.ArgumentParser(description="LlamaIndex RAG PDF QA Agent (Persistent & Smart Cache)")
    parser.add_argument("--pdf", "-p", required=True, help="Path to PDF file to load or query")
    parser.add_argument("--query", "-q", help="Ask a question about the document")
    parser.add_argument("--chat", "-c", help="Start a chat interaction")
    parser.add_argument("--run-eval", action="store_true", help="Run evaluation logic (optional)")
    parser.add_argument("--env", help="Path to .env file (optional)")
    parser.add_argument("--log-level", default="INFO", help="Logging level (INFO, DEBUG, etc.)")
    return parser.parse_args()


def compute_pdf_hash(pdf_path: str) -> str:
    """Compute a short MD5 hash of the PDF content."""
    with open(pdf_path, "rb") as f:
        data = f.read()
    return hashlib.md5(data).hexdigest()[:12]  # shorter hash for readability


def generate_collection_names(pdf_path: str, pdf_hash: str) -> tuple[str, str]:
    """Generate deterministic collection names for OpenAI and BGE embeddings."""
    return (
        f"openai_collection_{pdf_hash}",
        f"bge_collection_{pdf_hash}",
    )


def get_or_build_indices(pdf_path: str, env: dict):
    """
    Check if an index already exists in Chroma for this exact PDF.
    If found and hash matches → reuse.
    If hash differs → reindex and overwrite old collection.
    """
    chroma_path = env["CHROMA_DB_PATH"]
    openai_api_key = env["OPENAI_API_KEY"]

    db = chromadb.PersistentClient(path=chroma_path)
    existing_collections = [c.name for c in db.list_collections()]

    current_hash = compute_pdf_hash(pdf_path)
    openai_coll_name, bge_coll_name = generate_collection_names(pdf_path, current_hash)

    logging.info(f"📄 PDF hash for '{Path(pdf_path).name}': {current_hash}")

    # Find if there's any previous version of this PDF indexed (by name prefix)
    previous_versions = [c for c in existing_collections if Path(pdf_path).stem in c]

    # If both collections for this hash already exist → reuse
    if openai_coll_name in existing_collections and bge_coll_name in existing_collections:
        logging.info("✅ Existing up-to-date vector indices found — skipping rebuild.")
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.core import StorageContext, VectorStoreIndex

        openai_store = ChromaVectorStore(chroma_collection=db.get_collection(openai_coll_name))
        bge_store = ChromaVectorStore(chroma_collection=db.get_collection(bge_coll_name))

        openai_index = VectorStoreIndex.from_vector_store(vector_store=openai_store)
        bge_index = VectorStoreIndex.from_vector_store(vector_store=bge_store)
        return openai_index, bge_index

    # Otherwise, reindex
    logging.warning("⚠️ No matching or outdated vector stores found — rebuilding embeddings.")

    # Optionally, delete old collections related to this PDF
    for old in previous_versions:
        logging.info(f"🧹 Removing outdated collection: {old}")
        try:
            db.delete_collection(old)
        except Exception as e:
            logging.warning(f"Could not delete old collection {old}: {e}")

    # Rebuild from scratch
    documents = load_pdf_documents(pdf_path)
    nodes = split_documents_to_nodes(documents, openai_api_key=openai_api_key)

    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    openai_coll = db.get_or_create_collection(openai_coll_name)
    bge_coll = db.get_or_create_collection(bge_coll_name)

    openai_store = ChromaVectorStore(chroma_collection=openai_coll)
    bge_store = ChromaVectorStore(chroma_collection=bge_coll)

    storage_openai = StorageContext.from_defaults(vector_store=openai_store)
    storage_bge = StorageContext.from_defaults(vector_store=bge_store)

    openai_embed = OpenAIEmbedding(model="text-embedding-3-small", api_key=openai_api_key)
    hf_embed = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # Build OpenAI-based index
    openai_index = VectorStoreIndex.from_documents(
    nodes,
    storage_context=storage_openai,
    embed_model=openai_embed,  # 👈 explicitly use OpenAI
    )

    # Build HuggingFace-based index
    bge_index = VectorStoreIndex.from_documents(
    nodes,
    storage_context=storage_bge,
    embed_model=hf_embed,  # 👈 explicitly use HuggingFace
    )

    logging.info("✅ Rebuilt and saved fresh vector indices.")
    return openai_index, bge_index


def main():
    args = parse_args()
    setup_logging(getattr(logging, args.log_level.upper(), logging.INFO))
    env = load_env(args.env)

    pdf_path = args.pdf
    openai_index, bge_index = get_or_build_indices(pdf_path, env)

    # Query / Chat
    if args.query:
        logging.info("----- QUERY MODE -----")
        responses = query_index(openai_index, bge_index, args.query)
        print("\n🧠 OpenAI →", responses["openai"], "\n")
        print("🤗 HuggingFace →", responses["huggingface"], "\n")

    if args.chat:
        logging.info("----- CHAT MODE -----")
        print(f"\n🧠 OpenAI → {chat_index(openai_index, args.chat)}\n")
        print(f"🤗 HuggingFace → {chat_index(bge_index, args.chat)}\n")

    # Evaluation
    if args.run_eval:
        logging.info("----- RUNNING EVALUATION -----")
        evaluate_indices(openai_index, bge_index, openai_api_key=env["OPENAI_API_KEY"])

    logging.info("✅ Execution complete.")


if __name__ == "__main__":
    main()
