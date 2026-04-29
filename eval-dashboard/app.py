"""RAG evaluation dashboard for the Finance Insight Assistant.

A Streamlit page that runs RAGAS metrics (LLMContextRecall, Faithfulness,
FactualCorrectness) over a curated 22-question financial Q&A benchmark and
visualizes per-question and aggregate scores.
"""

import os

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from dataset import NUM_QUESTIONS, queries, expected_responses
from evaluator import run_eval

# ===================== Page Configuration =====================
st.set_page_config(
    page_title="Finance Insight Assistant — RAG Eval Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

ALL_METRICS = ["LLMContextRecall", "Faithfulness", "FactualCorrectness"]
METRIC_LABELS = {
    "LLMContextRecall": "Context Recall",
    "Faithfulness": "Faithfulness",
    "FactualCorrectness": "Factual Correctness",
}
METRIC_DESCRIPTIONS = {
    "LLMContextRecall": "Did the retrieved context contain the information needed to answer?",
    "Faithfulness": "Are the claims in the answer actually supported by the retrieved context?",
    "FactualCorrectness": "Does the answer agree with the reference answer on the facts?",
}

# ===================== Sidebar =====================
st.sidebar.title("⚙️ Run configuration")

n_questions = st.sidebar.slider(
    "Questions to evaluate",
    min_value=1,
    max_value=NUM_QUESTIONS,
    value=min(5, NUM_QUESTIONS),
    help="Smaller runs are cheaper and faster (~$0.005/question with gpt-4o-mini).",
)

selected_metrics = st.sidebar.multiselect(
    "Metrics",
    options=ALL_METRICS,
    default=ALL_METRICS,
    format_func=lambda m: METRIC_LABELS[m],
)

evaluator_model = st.sidebar.selectbox(
    "Evaluator LLM",
    options=["gpt-4o-mini", "gpt-4o"],
    index=0,
    help="Model used by RAGAS to score responses. gpt-4o-mini is ~10× cheaper.",
)

run_clicked = st.sidebar.button(
    "▶️ Run evaluation",
    type="primary",
    disabled=(len(selected_metrics) == 0),
    width='stretch',
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Each question makes one RAG call (api/chatbot) + several scoring calls "
    "(one per selected metric). Total cost scales with `questions × metrics`."
)

# ===================== Main =====================
st.title("📊 RAG Evaluation Dashboard")
st.markdown(
    "Measures retrieval quality and answer groundedness on a curated "
    f"**{NUM_QUESTIONS}-question financial Q&A benchmark**, using "
    "[RAGAS](https://docs.ragas.io). Pick a metric subset and a question "
    "count in the sidebar, then run."
)

# ---- Default view: dataset preview ----
if not run_clicked and "last_result" not in st.session_state:
    st.subheader("Benchmark preview")
    preview_df = pd.DataFrame({
        "Question": queries,
        "Expected answer (excerpt)": [
            (r[:140] + "…") if len(r) > 140 else r
            for r in expected_responses
        ],
    })
    st.dataframe(preview_df, width='stretch', height=400)

    st.subheader("What each metric measures")
    for m in ALL_METRICS:
        st.markdown(f"- **{METRIC_LABELS[m]}** — {METRIC_DESCRIPTIONS[m]}")

# ---- Run a fresh evaluation ----
if run_clicked:
    if not os.getenv("OPENAI_API_KEY"):
        st.error(
            "OPENAI_API_KEY is not set in this container's environment. "
            "Set it in eval-dashboard/.env (or root .env if shared) and restart."
        )
        st.stop()

    progress = st.progress(0.0, text=f"Querying RAG for {n_questions} questions…")

    def update_progress(done: int, total: int):
        progress.progress(done / total, text=f"Querying RAG: {done}/{total}…")

    with st.spinner(f"Scoring {n_questions} responses with RAGAS — this may take a minute…"):
        try:
            df = run_eval(
                n_questions=n_questions,
                metric_names=selected_metrics,
                evaluator_model=evaluator_model,
                progress_cb=update_progress,
            )
            st.session_state["last_result"] = df
            st.session_state["last_metrics"] = selected_metrics
        except Exception as e:
            st.error(f"Evaluation failed: {e}")
            st.stop()

    progress.empty()

# ---- Render the most recent result ----
if "last_result" in st.session_state:
    df: pd.DataFrame = st.session_state["last_result"]
    metrics_used = st.session_state["last_metrics"]

    st.subheader("Aggregate scores")
    agg_cols = st.columns(len(metrics_used))
    for col, metric in zip(agg_cols, metrics_used):
        # Column name in df can be the camelCase metric class name, or RAGAS's
        # snake_case label — defensively handle both.
        candidates = [metric, metric.lower(),
                      "context_recall" if metric == "LLMContextRecall" else None,
                      "faithfulness" if metric == "Faithfulness" else None,
                      "factual_correctness" if metric == "FactualCorrectness" else None,
                      "factual_correctness(mode=f1)" if metric == "FactualCorrectness" else None]
        col_name = next((c for c in candidates if c and c in df.columns), None)
        if col_name is None:
            col.metric(METRIC_LABELS[metric], "—")
        else:
            mean = pd.to_numeric(df[col_name], errors="coerce").mean()
            col.metric(METRIC_LABELS[metric], f"{mean:.3f}" if pd.notna(mean) else "—")

    # Per-question bar chart
    st.subheader("Per-question scores")
    score_cols = [c for c in df.columns
                  if c not in ("user_input", "retrieved_contexts", "response",
                               "reference", "sources", "error")
                  and pd.api.types.is_numeric_dtype(df[c])]
    if score_cols:
        chart_df = df[["user_input"] + score_cols].copy()
        chart_df["question_idx"] = range(len(chart_df))
        long_df = chart_df.melt(
            id_vars=["question_idx", "user_input"],
            value_vars=score_cols,
            var_name="metric",
            value_name="score",
        )
        fig = px.bar(
            long_df,
            x="question_idx",
            y="score",
            color="metric",
            barmode="group",
            hover_data=["user_input"],
            labels={"question_idx": "Question #", "score": "Score (0–1)"},
        )
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, width='stretch')

    # Drill-down table
    st.subheader("Per-question details")
    for i, row in df.iterrows():
        score_summary = " · ".join([
            f"{METRIC_LABELS.get(c, c)}: {row[c]:.2f}"
            for c in score_cols
            if pd.notna(row.get(c))
        ])
        header = f"**Q{i+1}.** {row['user_input']}  —  {score_summary}"
        with st.expander(header):
            if row.get("error"):
                st.error(f"Failed: {row['error']}")
            cols = st.columns(2)
            with cols[0]:
                st.markdown("**Generated answer**")
                st.write(row.get("response") or "_(empty)_")
                st.caption(f"Sources: {row.get('sources') or '—'}")
            with cols[1]:
                st.markdown("**Reference answer**")
                st.write(row.get("reference") or "_(empty)_")

            ctxs = row.get("retrieved_contexts")
            if isinstance(ctxs, list) and ctxs:
                st.markdown("**Retrieved chunks**")
                for j, ctx in enumerate(ctxs):
                    st.caption(f"Chunk {j+1}")
                    st.write(ctx[:600] + ("…" if len(ctx) > 600 else ""))

    # Download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download results as CSV",
        data=csv,
        file_name="rag_eval_results.csv",
        mime="text/csv",
    )
