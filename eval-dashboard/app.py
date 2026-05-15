"""RAG evaluation dashboard for the Finance Insight Assistant.

A Streamlit page that runs RAGAS metrics (LLMContextRecall, Faithfulness,
FactualCorrectness) over a curated 22-question financial Q&A benchmark and
visualizes per-question and aggregate scores. Supports comparing hybrid vs.
dense-only retrieval side-by-side and shows per-chunk provenance badges.
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

# Streamlit renders `$...$` as LaTeX math, mangling dollar amounts in answers.
def _safe(text) -> str:
    return text.replace("$", "\\$") if isinstance(text, str) else text


def _metric_col(metric: str, df: pd.DataFrame):
    """Return the DataFrame column name for a RAGAS metric, or None."""
    candidates = [
        metric,
        metric.lower(),
        "context_recall"                   if metric == "LLMContextRecall"    else None,
        "llm_context_precision_with_reference" if metric == "LLMContextPrecision" else None,
        "llm_context_precision"                if metric == "LLMContextPrecision" else None,
        "context_precision"                    if metric == "LLMContextPrecision" else None,
        "faithfulness"                     if metric == "Faithfulness"         else None,
        "factual_correctness"              if metric == "FactualCorrectness"   else None,
        "factual_correctness(mode=f1)"     if metric == "FactualCorrectness"   else None,
        "factual_correctness(mode=recall)" if metric == "FactualCorrectness"   else None,
    ]
    return next((c for c in candidates if c and c in df.columns), None)


def _retrieval_badge(prov: dict) -> str:
    """Return an HTML badge string for a chunk's retrieval provenance."""
    if not prov:
        return ""
    src = prov.get("source") or ""
    dense_r = prov.get("dense_rank")
    bm25_r  = prov.get("bm25_rank")
    rrf     = prov.get("rrf_score")
    rrf_str = f"{rrf:.4f}" if rrf is not None else "?"
    dense_r_str = f"#{dense_r}" if dense_r is not None else "#?"
    bm25_r_str  = f"#{bm25_r}"  if bm25_r  is not None else "#?"
    base = "border-radius:4px;padding:2px 7px;font-size:0.78em;margin-left:6px;"
    if src == "both":
        return (
            f"<span style='background:#2d6a4f;color:white;{base}'>"
            f"Both &nbsp;·&nbsp; Dense {dense_r_str} &nbsp;·&nbsp; BM25 {bm25_r_str} &nbsp;·&nbsp; RRF {rrf_str}"
            f"</span>"
        )
    if src == "dense":
        return (
            f"<span style='background:#023e8a;color:white;{base}'>"
            f"Dense {dense_r_str} &nbsp;·&nbsp; Score {rrf_str}"
            f"</span>"
        )
    if src == "bm25":
        return (
            f"<span style='background:#e85d04;color:white;{base}'>"
            f"BM25 {bm25_r_str} &nbsp;·&nbsp; Score {rrf_str}"
            f"</span>"
        )
    return ""


def _source_summary(provenance_list: list) -> str:
    """Return a human-readable summary of chunk sources, e.g. '5 dense-only · 3 both'."""
    counts = {"dense": 0, "bm25": 0, "both": 0}
    for p in provenance_list:
        src = (p.get("source") or "").lower()
        if src in counts:
            counts[src] += 1
    parts = []
    if counts["dense"]:
        parts.append(f"{counts['dense']} dense-only")
    if counts["bm25"]:
        parts.append(f"{counts['bm25']} BM25-only")
    if counts["both"]:
        parts.append(f"{counts['both']} both")
    return " · ".join(parts) if parts else "no provenance data"


_SCORE_COLS_EXCLUDE = frozenset({
    "user_input", "retrieved_contexts", "response",
    "reference", "sources", "provenance", "error",
})

def _score_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns
            if c not in _SCORE_COLS_EXCLUDE
            and pd.api.types.is_numeric_dtype(df[c])]


ALL_METRICS = ["LLMContextRecall", "LLMContextPrecision", "Faithfulness", "FactualCorrectness"]
METRIC_LABELS = {
    "LLMContextRecall":    "Context Recall",
    "LLMContextPrecision": "Context Precision",
    "Faithfulness":         "Faithfulness",
    "FactualCorrectness":   "Factual Correctness",
}
METRIC_DESCRIPTIONS = {
    "LLMContextRecall":    "Did the retrieved context contain the information needed to answer?",
    "LLMContextPrecision": "Are the retrieved chunks relevant, or is there noise? (are we retrieving too much garbage?)",
    "Faithfulness":         "Are the claims in the answer actually supported by the retrieved context?",
    "FactualCorrectness":   "Does the answer cover the facts in the reference? (recall mode — extra correct detail is not penalised)",
}

MODE_OPTIONS = ["hybrid", "dense"]  # sparse-only mode intentionally excluded from eval comparison
MODE_LABELS  = {"hybrid": "Hybrid (Dense + BM25)", "dense": "Dense-only"}

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

selected_modes = st.sidebar.multiselect(
    "Retrieval modes",
    options=MODE_OPTIONS,
    default=MODE_OPTIONS,
    format_func=lambda m: MODE_LABELS[m],
    help="Select one mode for a standard run, or both to compare side-by-side.",
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
    disabled=(len(selected_metrics) == 0 or len(selected_modes) == 0),
    width='stretch',
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Each question makes one RAG call per retrieval mode (api/chatbot) + several "
    "scoring calls (one per selected metric). Total cost scales with "
    "`questions × metrics × modes`."
)

# ===================== Main =====================
st.title("📊 RAG Evaluation Dashboard")
st.markdown(
    "Measures retrieval quality and answer groundedness on a curated "
    f"**{NUM_QUESTIONS}-question financial Q&A benchmark**, using "
    "[RAGAS](https://docs.ragas.io). Pick metrics, retrieval modes, and a question "
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

    n_passes = len(selected_modes)
    total_rag_calls = n_questions * n_passes
    progress = st.progress(0.0, text=f"Querying RAG: 0/{total_rag_calls}…")

    def update_progress(done: int, total: int):
        progress.progress(done / total, text=f"Querying RAG: {done}/{total}…")

    spinner_text = (
        f"Scoring {total_rag_calls} responses "
        f"({n_passes} mode{'s' if n_passes > 1 else ''}) with RAGAS — "
        "this may take a minute…"
    )
    with st.spinner(spinner_text):
        try:
            result_dict = run_eval(
                n_questions=n_questions,
                metric_names=selected_metrics,
                evaluator_model=evaluator_model,
                retrieval_modes=selected_modes,
                progress_cb=update_progress,
            )
            st.session_state["last_result"]  = result_dict
            st.session_state["last_metrics"] = selected_metrics
            st.session_state["last_modes"]   = selected_modes
        except Exception as e:
            st.error(f"Evaluation failed: {e}")
            st.stop()

    progress.empty()

# ---- Render the most recent result ----
if "last_result" in st.session_state:
    result_dict: dict = st.session_state["last_result"]
    metrics_used: list = st.session_state["last_metrics"]
    modes_used: list   = st.session_state.get("last_modes", ["hybrid"])

    # ---- Aggregate scores ----
    st.subheader("Aggregate scores")

    agg_rows = []
    mode_means: dict = {}
    for mode in modes_used:
        df = result_dict[mode]
        mode_means[mode] = {}
        for metric in metrics_used:
            col_name = _metric_col(metric, df)
            mean = (
                pd.to_numeric(df[col_name], errors="coerce").mean()
                if col_name else float("nan")
            )
            mode_means[mode][metric] = mean
            agg_rows.append({
                "metric": METRIC_LABELS[metric],
                "mode":   MODE_LABELS.get(mode, mode),
                "score":  mean,
            })

    agg_df = pd.DataFrame(agg_rows)
    color_map = {
        MODE_LABELS["hybrid"]: "#4C78A8",
        MODE_LABELS["dense"]:  "#F58518",
    }
    fig = px.bar(
        agg_df,
        x="metric",
        y="score",
        color="mode",
        barmode="group",
        labels={"metric": "Metric", "score": "Score (0–1)", "mode": "Retrieval mode"},
        color_discrete_map=color_map,
    )
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    # Delta chips (hybrid − dense)
    if "hybrid" in modes_used and "dense" in modes_used:
        delta_cols = st.columns(len(metrics_used))
        for col, metric in zip(delta_cols, metrics_used):
            h = mode_means.get("hybrid", {}).get(metric, float("nan"))
            d = mode_means.get("dense",  {}).get(metric, float("nan"))
            if pd.notna(h) and pd.notna(d):
                delta = h - d
                sign  = "+" if delta >= 0 else ""
                color = "green" if delta >= 0 else "red"
                col.markdown(
                    f"<div style='text-align:center'>"
                    f"<span style='color:{color};font-size:0.85em;'>"
                    f"{sign}{delta:.3f} vs dense-only</span></div>",
                    unsafe_allow_html=True,
                )

    # ---- Metric definitions ----
    st.subheader("What each metric measures")
    for m in metrics_used:
        st.markdown(f"- **{METRIC_LABELS[m]}** — {METRIC_DESCRIPTIONS[m]}")

    # ---- Per-question scores chart ----
    st.subheader("Per-question scores")

    if len(modes_used) > 1:
        chart_mode = st.radio(
            "Show mode",
            options=modes_used + ["both"],
            format_func=lambda m: {**MODE_LABELS, "both": "Both modes"}.get(m, m),
            horizontal=True,
            key="chart_mode_radio",
        )
        chart_modes_to_render = modes_used if chart_mode == "both" else [chart_mode]
    else:
        chart_modes_to_render = modes_used

    long_rows = []
    for mode in chart_modes_to_render:
        df = result_dict[mode]
        score_cols = _score_cols(df)
        if score_cols:
            chart_df = df[["user_input"] + score_cols].copy()
            chart_df["question_idx"] = range(1, len(chart_df) + 1)
            ldf = chart_df.melt(
                id_vars=["question_idx", "user_input"],
                value_vars=score_cols,
                var_name="metric",
                value_name="score",
            )
            ldf["mode"] = MODE_LABELS.get(mode, mode)
            long_rows.append(ldf)

    if long_rows:
        long_df = pd.concat(long_rows, ignore_index=True)
        long_df["metric_mode"] = long_df["metric"] + " (" + long_df["mode"] + ")"
        fig = px.bar(
            long_df,
            x="question_idx",
            y="score",
            color="metric_mode",
            barmode="group",
            hover_data=["user_input"],
            labels={"question_idx": "Question #", "score": "Score (0–1)"},
        )
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    # ---- Per-question details ----
    st.subheader("Per-question details")

    if len(modes_used) > 1:
        detail_mode = st.radio(
            "Show details for",
            options=modes_used,
            format_func=lambda m: MODE_LABELS.get(m, m),
            horizontal=True,
            key="detail_mode_radio",
        )
    else:
        detail_mode = modes_used[0]

    df = result_dict[detail_mode]
    score_cols = _score_cols(df)

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
                st.write(_safe(row.get("response")) or "_(empty)_")
                st.caption(f"Sources: {row.get('sources') or '—'}")
            with cols[1]:
                st.markdown("**Reference answer**")
                st.write(_safe(row.get("reference")) or "_(empty)_")

            ctxs = row.get("retrieved_contexts")
            provenance_list = row.get("provenance") or []
            if isinstance(ctxs, list) and ctxs:
                st.markdown("**Retrieved chunks**")
                if provenance_list:
                    st.caption(_source_summary(provenance_list))
                for j, ctx in enumerate(ctxs):
                    prov  = provenance_list[j] if j < len(provenance_list) else {}
                    badge = _retrieval_badge(prov)
                    st.markdown(
                        f"Chunk {j + 1} &nbsp;{badge}",
                        unsafe_allow_html=True,
                    )
                    st.write(_safe(ctx[:600] + ("…" if len(ctx) > 600 else "")))

    # ---- Download ----
    st.markdown("---")
    dl_cols = st.columns(len(modes_used))
    for col, mode in zip(dl_cols, modes_used):
        df = result_dict[mode]
        csv = df.to_csv(index=False).encode("utf-8")
        col.download_button(
            f"⬇️ Download {MODE_LABELS.get(mode, mode)} CSV",
            data=csv,
            file_name=f"rag_eval_results_{mode}.csv",
            mime="text/csv",
        )
