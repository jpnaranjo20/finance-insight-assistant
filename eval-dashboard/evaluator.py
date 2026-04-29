"""RAGAS evaluation pipeline against the api/ RAG endpoint.

This module exposes one entry point — `run_eval(...)` — that:
  1. Sends each query in dataset.queries to the api service's /chatbot endpoint.
  2. Captures the LLM response and the retrieved docs.
  3. Runs RAGAS metrics (LLMContextRecall, Faithfulness, FactualCorrectness)
     against the (query, response, contexts, reference) tuples.
  4. Returns a pandas DataFrame with per-row scores plus the aggregate means.

The api service is reached at http://api:80/chatbot when this dashboard is
running inside the docker-compose network. Override via the API_URL env var.
"""

import os
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests

from dataset import queries, expected_responses

API_URL = os.getenv("API_URL", "http://api:80/chatbot")
DEFAULT_TIMEOUT = 120  # seconds; first call can be slow if api just started


def fetch_rag_response(query: str) -> Tuple[str, List[str], List[str]]:
    """Call the RAG endpoint once and return (answer, contexts, sources)."""
    resp = requests.post(API_URL, json={"question": query}, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    answer = data.get("llm_response", "") or ""
    retrieved = data.get("retrieved_docs") or []

    contexts: List[str] = []
    sources: List[str] = []
    for doc in retrieved:
        contexts.append(doc.get("page_content", "") or "")
        meta = doc.get("metadata") or {}
        src = meta.get("source")
        if src and src not in sources:
            sources.append(src)

    return answer, contexts, sources


def build_eval_records(
    n_questions: int,
    progress_cb=None,
) -> List[Dict[str, Any]]:
    """Hit the RAG API for each of the first `n_questions` and assemble the
    records RAGAS needs."""
    records: List[Dict[str, Any]] = []
    for i in range(n_questions):
        q = queries[i]
        ref = expected_responses[i]
        try:
            answer, contexts, sources = fetch_rag_response(q)
            error = None
        except Exception as e:
            answer = ""
            contexts = []
            sources = []
            error = str(e)

        records.append({
            "user_input": q,
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": ref,
            "sources": sources,
            "error": error,
        })

        if progress_cb is not None:
            progress_cb(i + 1, n_questions)

    return records


def run_eval(
    n_questions: int,
    metric_names: List[str],
    evaluator_model: str = "gpt-4o-mini",
    progress_cb=None,
) -> pd.DataFrame:
    """End-to-end: query the RAG API for the first N questions, then run
    RAGAS with the requested metrics. Returns one row per question with
    the user input, retrieved sources, answer, reference, and per-metric
    scores."""

    # Heavy ragas + langchain imports go here so module import (and thus the
    # Streamlit page render) stays fast.
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        FactualCorrectness,
        LLMContextRecall,
    )

    records = build_eval_records(n_questions, progress_cb=progress_cb)

    # Drop rows where the API call failed — RAGAS would crash on empty
    # contexts / response. We keep a record of failures separately so the UI
    # can surface them.
    valid_records = [r for r in records if r["error"] is None and r["response"]]

    metric_map = {
        "LLMContextRecall": LLMContextRecall(),
        "Faithfulness": Faithfulness(),
        "FactualCorrectness": FactualCorrectness(),
    }
    metrics = [metric_map[name] for name in metric_names if name in metric_map]
    if not metrics:
        raise ValueError(f"No valid metrics selected from: {metric_names}")

    ragas_dataset = EvaluationDataset.from_list([
        {
            "user_input": r["user_input"],
            "retrieved_contexts": r["retrieved_contexts"],
            "response": r["response"],
            "reference": r["reference"],
        }
        for r in valid_records
    ])

    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(model=evaluator_model, temperature=0)
    )

    result = evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
        llm=evaluator_llm,
    )

    # `result` exposes a to_pandas() that gives per-row scores. Merge it back
    # with the source/error info so the UI has everything in one frame.
    scores_df = result.to_pandas()

    # Reattach metadata columns by index alignment. RAGAS preserves order.
    metadata_df = pd.DataFrame([
        {
            "sources": ", ".join(r["sources"]),
            "error": r["error"],
        }
        for r in valid_records
    ])
    full_df = pd.concat([scores_df.reset_index(drop=True),
                         metadata_df.reset_index(drop=True)], axis=1)

    # Append failure rows at the end with NaN scores so the user sees them.
    failed_records = [r for r in records if r["error"] is not None or not r["response"]]
    if failed_records:
        failed_df = pd.DataFrame([
            {
                "user_input": r["user_input"],
                "retrieved_contexts": r["retrieved_contexts"],
                "response": r["response"],
                "reference": r["reference"],
                "sources": ", ".join(r["sources"]),
                "error": r["error"] or "empty response",
            }
            for r in failed_records
        ])
        full_df = pd.concat([full_df, failed_df], ignore_index=True)

    return full_df
