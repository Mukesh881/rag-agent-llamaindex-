"""
Query and chat interfaces for the RAG agent.
"""

import os
import logging
from llama_index.core import get_response_synthesizer


def make_query_engine(index, mode: str = "compact"):
    """Return a query engine configured with response synthesizer."""
    synthesizer = get_response_synthesizer(response_mode=mode)
    return index.as_query_engine(response_synthesizer=synthesizer)


def query_index(openai_index, hf_index, question: str, response_mode: str = "compact") -> dict[str, str]:
    """
    Query both OpenAI-based and HuggingFace-based indices with the same question.

    Args:
        openai_index: VectorStoreIndex built using OpenAI embeddings.
        hf_index: VectorStoreIndex built using HuggingFace embeddings.
        question: User's natural language query.
        response_mode: Response synthesizer mode (e.g., "compact", "tree_summarize").

    Returns:
        dict[str, str]: Dictionary with responses from both query engines.
    """
    from llama_index.core import get_response_synthesizer
    from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
    from llama_index.llms.openai import OpenAI

    logging.info(f"Running query across both OpenAI and HuggingFace indices: {question}")

    # --- Response synthesizers ---
    synth = get_response_synthesizer(response_mode=response_mode)

    # --- Query engine 1: OpenAI ---
    openai_llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
    q_engine_openai = openai_index.as_query_engine(
        llm=openai_llm,
        response_synthesizer=synth
    )

    # --- Query engine 2: HuggingFace ---
    temperature = max(1e-5, float(os.getenv("HF_TEMPERATURE", 0.0)))

    hf_llm = HuggingFaceInferenceAPI(
        model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",  # you can swap to any hosted model
        token= os.getenv("HUGGINGFACE_API_KEY"),
        temperature=temperature,
        max_new_tokens=512,
    )
    q_engine_hf = hf_index.as_query_engine(
        llm=hf_llm,
        response_synthesizer=synth
    )

    # --- Run both queries ---
    logging.info("Querying OpenAI engine...")
    openai_response = q_engine_openai.query(question)
    logging.info("Querying HuggingFace engine...")
    hf_response = q_engine_hf.query(question)

    results = {
        "openai": getattr(openai_response, "response", str(openai_response)),
        "huggingface": getattr(hf_response, "response", str(hf_response)),
    }

    logging.info("✅ Query completed for both engines.")
    return results



def chat_index(index, message: str, mode: str = "compact") -> str:
    """Chat interactively with the index."""
    synthesizer = get_response_synthesizer(response_mode=mode)
    chat_engine = index.as_chat_engine(response_synthesizer=synthesizer)
    logging.info(f"Chat: {message}")
    response = chat_engine.chat(message)
    return getattr(response, "response", str(response))