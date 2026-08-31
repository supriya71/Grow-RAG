"""HDFC Fund FAQ — Streamlit single page (Phase 7: retrieval + generation).

Run from the repo root:
    <venv>/python -m streamlit run code/ui/app.py

Wires the PRD UI (welcome, 3 examples, facts-only disclaimer, chat input,
answer + one citation link + last-updated) to a RAG loop: retrieve top-k
chunks, then synthesize a grounded factual answer via generation.generator.
"""

from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import streamlit as st

from config.corpus import CORPUS

DEFAULT_K = 5

EXAMPLES = [
    "What is the expense ratio of HDFC Large Cap Fund Direct Growth?",
    "What is the lock-in for HDFC ELSS Tax Saver?",
    "What is the minimum SIP and exit load for HDFC Small Cap Fund Direct Growth?",
]


def run_flow(question: str) -> None:
    from retrieval.retriever import retrieve
    from generation.generator import generate

    with st.spinner("Retrieving evidence and drafting the answer..."):
        evidence = retrieve(question, k=DEFAULT_K)
        result = generate(question, evidence)

    st.session_state["result"] = result


def render_result(result: dict) -> None:
    st.markdown(f"### {result['query']}")

    policy = result.get("policy", "answer")
    if policy in ("advice", "returns"):
        st.info(result["answer"])
    elif policy == "pii":
        st.warning(result["answer"])
    elif result.get("empty"):
        st.info(result["answer"])
    else:
        st.success(result["answer"])

    col_cite, col_ts = st.columns([3, 2])
    if result.get("citation_url"):
        col_cite.markdown(
            f"**Source:** [{result['citation_label']}]({result['citation_url']})"
        )
    else:
        col_cite.markdown("**Source:** none cited")
    ts = result.get("last_updated")
    col_ts.markdown(
        f"**Last updated from sources:** `{ts}`" if ts else "**Last updated from sources:** —"
    )


st.set_page_config(page_title="HDFC Fund FAQ", layout="centered")

st.markdown(
    """
    <style>
      .kicker { color: #00664A; letter-spacing: .08em; font-size: .75rem; font-weight: 600; }
      .disclaimer {
        background: #E8F5F0; border: 1px solid #00B386; border-radius: 8px;
        padding: 8px 12px; color: #0E2A24; margin-bottom: 4px;
      }
      a { color: #00B386; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="kicker">GROWW HDFC FAQ PROTOTYPE</p>', unsafe_allow_html=True)
st.title("Mutual fund facts")
st.markdown(
    '<div class="disclaimer">Facts-only. No investment advice.</div>',
    unsafe_allow_html=True,
)

st.subheader("Covered funds (5)")
for entry in CORPUS:
    st.markdown(f"- **{entry['fund_name']}** — [Groww page]({entry['url']})")

st.divider()
st.caption("Try one of these:")

cols = st.columns(len(EXAMPLES))
picked: str | None = None
for col, example in zip(cols, EXAMPLES):
    if col.button(example):
        picked = example
if picked is not None:
    st.session_state.ask_box = picked

with st.form("ask"):
    query_text = st.text_input(
        "Ask a factual question about the five funds",
        key="ask_box",
        placeholder="e.g. What is the lock-in for HDFC ELSS Tax Saver?",
    )
    submitted = st.form_submit_button("Ask")

question = query_text if submitted else picked

if question:
    question = question.strip()
    if not question:
        st.warning("Ask a factual question about the five HDFC funds.")
    else:
        st.session_state.last_query = question
        try:
            run_flow(question)
        except RuntimeError as exc:
            st.error(str(exc))
            st.session_state.pop("result", None)

st.divider()
result = st.session_state.get("result")
if result is None:
    st.info("Ask a factual question above; you'll get a grounded answer with a source link.")
else:
    render_result(result)

if st.button("Clear results"):
    st.session_state.pop("result", None)
    st.session_state.pop("ask_box", None)
    st.session_state.pop("last_query", None)
    st.rerun()
