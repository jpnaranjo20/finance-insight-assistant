"""RAGAS evaluation pipeline against the api/ RAG endpoint.

This module exposes one entry point — `run_eval(...)` — that:
  1. For each retrieval mode, sends each query to the api service's /chatbot endpoint.
  2. Captures the LLM response, retrieved docs, and per-chunk provenance.
  3. Runs RAGAS metrics against (query, response, contexts, reference) tuples.
  4. Returns a dict[str, pd.DataFrame] keyed by retrieval mode name.

The api service is reached at http://api:80/chatbot when this dashboard is
running inside the docker-compose network. Override via the API_URL env var.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from dataset import queries, expected_responses

API_URL = os.getenv("API_URL", "http://api:80/chatbot")
DEFAULT_TIMEOUT = 120  # seconds; first call can be slow if api just started


def fetch_rag_response(
    query: str,
    retrieval_mode: str = "hybrid",
) -> Tuple[str, List[str], List[str], List[Dict[str, Any]]]:
    """Call the RAG endpoint once and return (answer, contexts, sources, provenance_list)."""
    resp = requests.post(
        API_URL,
        json={"question": query, "retrieval_mode": retrieval_mode},
        timeout=DEFAULT_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()

    answer = data.get("llm_response", "") or ""
    retrieved = data.get("retrieved_docs") or []

    contexts: List[str] = []
    sources: List[str] = []
    provenance_list: List[Dict[str, Any]] = []

    for doc in retrieved:
        contexts.append(doc.get("page_content", "") or "")
        meta = doc.get("metadata") or {}
        src = meta.get("source")
        if src and src not in sources:
            sources.append(src)
        provenance_list.append({
            "source":     meta.get("_retrieval_source"),
            "dense_rank": meta.get("_dense_rank"),
            "bm25_rank":  meta.get("_bm25_rank"),
            "rrf_score":  meta.get("_rrf_score"),
        })

    return answer, contexts, sources, provenance_list


def build_eval_records(
    n_questions: int,
    retrieval_mode: str = "hybrid",
    progress_cb=None,
) -> List[Dict[str, Any]]:
    """Hit the RAG API for each of the first `n_questions` and assemble the
    records RAGAS needs. Each record includes a `provenance` list."""
    records: List[Dict[str, Any]] = []
    for i in range(n_questions):
        q = queries[i]
        ref = expected_responses[i]
        try:
            answer, contexts, sources, provenance = fetch_rag_response(q, retrieval_mode)
            error = None
        except Exception as e:
            answer = ""
            contexts = []
            sources = []
            provenance = []
            error = str(e)

        records.append({
            "user_input":        q,
            "retrieved_contexts": contexts,
            "response":          answer,
            "reference":         ref,
            "sources":           sources,
            "provenance":        provenance,
            "error":             error,
        })

        if progress_cb is not None:
            progress_cb(i + 1, n_questions)

    return records


def run_eval(
    n_questions: int,
    metric_names: List[str],
    evaluator_model: str = "gpt-4o-mini",
    retrieval_modes: Optional[List[str]] = None,
    progress_cb=None,
) -> Dict[str, pd.DataFrame]:
    """For each retrieval mode, query the RAG API for the first N questions, then
    run RAGAS with the requested metrics. Returns dict[mode, DataFrame] — one row
    per question with user input, retrieved sources, answer, reference, provenance,
    and per-metric scores.

    Progress callback receives (done, total) where total = n_questions × len(modes).
    """

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

    if retrieval_modes is None:
        retrieval_modes = ["hybrid"]

    metric_map = {
        "LLMContextRecall": LLMContextRecall,
        "Faithfulness":      Faithfulness,
        "FactualCorrectness": FactualCorrectness,
    }
    metric_names_valid = [name for name in metric_names if name in metric_map]
    if not metric_names_valid:
        raise ValueError(f"No valid metrics selected from: {metric_names}")

    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(model=evaluator_model, temperature=0)
    )

    total_ticks = n_questions * len(retrieval_modes)

    def _make_progress_cb(offset: int):
        def inner(done: int, _total: int):
            if progress_cb is not None:
                progress_cb(offset + done, total_ticks)
        return inner

    results: Dict[str, pd.DataFrame] = {}

    for idx, mode in enumerate(retrieval_modes):
        records = build_eval_records(
            n_questions,
            retrieval_mode=mode,
            progress_cb=_make_progress_cb(idx * n_questions),
        )

        metrics = [metric_map[name]() for name in metric_names_valid]

        valid_records = [r for r in records if r["error"] is None and r["response"]]

        if not valid_records:
            results[mode] = pd.DataFrame([
                {
                    "user_input": r["user_input"],
                    "reference":  r["reference"],
                    "sources":    ", ".join(r["sources"]),
                    "provenance": r["provenance"],
                    "error":      r["error"] or "empty response",
                }
                for r in records
            ])
            continue

        ragas_dataset = EvaluationDataset.from_list([
            {
                "user_input":         r["user_input"],
                "retrieved_contexts": r["retrieved_contexts"],
                "response":           r["response"],
                "reference":          r["reference"],
            }
            for r in valid_records
        ])

        result = evaluate(
            dataset=ragas_dataset,
            metrics=metrics,
            llm=evaluator_llm,
        )

        scores_df = result.to_pandas()

        metadata_df = pd.DataFrame([
            {
                "sources":   ", ".join(r["sources"]),
                "provenance": r["provenance"],
                "error":     r["error"],
            }
            for r in valid_records
        ])
        full_df = pd.concat(
            [scores_df.reset_index(drop=True), metadata_df.reset_index(drop=True)],
            axis=1,
        )

        failed_records = [r for r in records if r["error"] is not None or not r["response"]]
        if failed_records:
            failed_df = pd.DataFrame([
                {
                    "user_input":         r["user_input"],
                    "retrieved_contexts": r["retrieved_contexts"],
                    "response":           r["response"],
                    "reference":          r["reference"],
                    "sources":            ", ".join(r["sources"]),
                    "provenance":         r["provenance"],
                    "error":              r["error"] or "empty response",
                }
                for r in failed_records
            ])
            full_df = pd.concat([full_df, failed_df], ignore_index=True)

        results[mode] = full_df

    return results
