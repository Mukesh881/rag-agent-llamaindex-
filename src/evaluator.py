"""
Evaluation utilities for RAG responses (faithfulness, relevancy, correctness).
Now with live on-screen reporting.
"""

import logging
from typing import List, Dict
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
    BatchEvalRunner,
)
from llama_index.core.llama_dataset.generator import RagDatasetGenerator


def evaluate_indices(openai_index, bge_index, openai_api_key: str | None = None):
    """
    Evaluate both OpenAI- and HuggingFace-based indices for faithfulness,
    relevancy, and correctness.

    Prints results directly to console.

    Args:
        openai_index: VectorStoreIndex built with OpenAI embeddings.
        bge_index: VectorStoreIndex built with HuggingFace embeddings.
        openai_api_key: OpenAI API key for evaluation LLM (default: GPT-3.5-turbo).
    """
    logging.info("===== Starting Evaluation =====")

    try:
        # Create LLM for evaluation
        eval_llm = OpenAI(temperature=0, model="gpt-3.5-turbo", api_key=openai_api_key)

        # Generate evaluation dataset from the documents in one index (same docs across both)
        logging.info("Generating synthetic evaluation dataset...")
        nodes = openai_index.docstore.docs.values()
        nodes = list(openai_index.docstore.docs.values())
        print(f"🧾 Found {len(nodes)} nodes in index docstore.")

        # Try to recover documents directly from the index; fallback to query-based generation
        try:
            nodes = list(openai_index.docstore.docs.values())
            if not nodes:
                raise ValueError("Index docstore empty, cannot generate synthetic dataset.")
        except Exception:
            logging.warning("No documents in index — generating dataset via ad-hoc queries.")
            # Use ad-hoc random queries (simple heuristic)
            sample_queries = [
                "Summarize the main topic of this document.",
                "What is the key benefit mentioned?",
                "List any numerical values discussed.",
                "What section describes eligibility or conditions?"
            ]
            eval_questions = sample_queries
            eval_answers = ["N/A"] * len(sample_queries)
        else:
            logging.info(f"Generating synthetic dataset from {len(nodes)} document nodes.")
            generator = RagDatasetGenerator.from_documents(list(nodes), llm=eval_llm, num_questions_per_chunk=2)
            eval_dataset = generator.generate_dataset_from_nodes()
            eval_questions = [ex.query for ex in eval_dataset.examples]
            eval_answers = [ex.reference_answer for ex in eval_dataset.examples]


        # Create evaluators
        faithfulness = FaithfulnessEvaluator(llm=eval_llm)
        relevancy = RelevancyEvaluator(llm=eval_llm)
        correctness = CorrectnessEvaluator(llm=eval_llm)

        # Set up batch runner for parallel evaluations
        runner = BatchEvalRunner(
            {
                "faithfulness": faithfulness,
                "relevancy": relevancy,
                "correctness": correctness,
            },
            workers=4,
        )

        # Evaluate OpenAI-based index
        logging.info("Evaluating OpenAI index...")
        results_openai = runner.evaluate_queries(
            openai_index.as_query_engine(),
            queries=eval_questions,
            reference=eval_answers,
        )

        # Evaluate HuggingFace-based index
        logging.info("Evaluating HuggingFace index...")
        results_bge = runner.evaluate_queries(
            bge_index.as_query_engine(),
            queries=eval_questions,
            reference=eval_answers,
        )

        # Display summary results
        print("\n" + "=" * 60)
        print("📊  EVALUATION RESULTS SUMMARY")
        print("=" * 60)

        def summarize(results: Dict[str, List]) -> Dict[str, float]:
            """Compute simple pass ratios for each metric."""
            summary = {}
            for key, value_list in results.items():
                passes = sum(r.passing for r in value_list)
                summary[key] = passes / len(value_list) if value_list else 0.0
            return summary

        summary_openai = summarize(results_openai)
        summary_bge = summarize(results_bge)

        print("\n🧠 OpenAI-based Index:")
        print("-" * 40)
        for metric, score in summary_openai.items():
            print(f"{metric.title():<15}: {score * 100:.1f}%")

        print("\n🤗 HuggingFace-based Index:")
        print("-" * 40)
        for metric, score in summary_bge.items():
            print(f"{metric.title():<15}: {score * 100:.1f}%")

        print("\nDetailed individual results:")
        for engine, results in [("OpenAI", results_openai), ("HuggingFace", results_bge)]:
            print(f"\n🔹 {engine} Responses")
            for metric, value_list in results.items():
                passing = sum(r.passing for r in value_list)
                print(f"  {metric.title():<12}: {passing}/{len(value_list)} passed")

        print("\n✅ Evaluation complete.\n")

    except Exception as exc:
        logging.exception("Evaluation failed: %s", exc)
        print(f"❌ Evaluation failed: {exc}")
