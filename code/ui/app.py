"""Retrieval explorer UI (Phase 5) — Streamlit single page.

Run from the repo root:
    py -3 -m streamlit run code/ui/app.py

Wires the PRD UI shell (welcome, 3 examples, facts-only note, chat input,
citation URL + last-updated) to `retrieval.retriever.retrieve`. Generation
(Mistral) is not wired yet, so the page shows the chunks a generator would be
allowed to cite, not a synthesized answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import streamlit as st

from config.corpus import CORPUS

FUND_NAME_BY_URL = {entry["url"]: entry["fund_name"] for entry in CORPUS}

DEFAULT_K = 5

EXAMPLES = [
    "What is the expense ratio of HDFC Large Cap Fund Direct Growth?",
    "What is the lock-in for HDFC ELSS Tax Saver?",
    "What is the minimum SIP and exit load for HDFC Small Cap Fund Direct Growth?",
]


def render_evidence(res: dict) -> None:
    """Render retrieved evidence: metadata, citation, last-updated, top-k chunks."""
    st.markdown(f"### {res['query']}")

    if res["empty"]:
        st.error(
            "No close match in the five indexed Groww pages (honest miss). "
            "A generator must say it does not have this rather than invent it."
        )
        return

    if res["matched_funds"]:
        names = ", ".join(FUND_NAME_BY_URL.get(u, u) for u in res["matched_funds"])
        st.caption(f"Detected fund(s): {names}")
    else:
        st.caption("No single fund named - ambiguous or no fund mention; evidence may mix funds.")

    if res["citation_urls"]:
        links = "  |  ".join(
            f"[{FUND_NAME_BY_URL.get(u, u)}]({u})" for u in res["citation_urls"]
        )
        st.markdown(f"**Citation URL(s):** {links}")

    st.markdown(f"**Last updated from sources:** `{res['fetched_at_max']}`")

    for rank, chunk in enumerate(res["chunks"], 1):
        similarity = min(max(1.0 - chunk["distance"], 0.0), 1.0)
        with st.container(border=True):
            head, score = st.columns([4, 1])
            head.markdown(f"**#{rank}** - {chunk['fund_name']}")
            score.markdown(f"`{chunk['distance']:.4f}` dist")
            st.markdown(f"[{chunk['url']}]({chunk['url']})")
            st.caption(f"{chunk['chunk_id']} - fetched {chunk['fetched_at']}")
            st.progress(similarity, text=f"similarity {similarity:.1%}")
            with st.expander("View chunk text"):
                st.text(chunk["text"])


st.set_page_config(page_title="HDFC Fund FAQ - Retrieval Explorer", layout="centered")

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
st.title("Mutual fund facts - retrieval explorer")
st.markdown(
    '<div class="disclaimer">Facts-only. No investment advice.</div>',
    unsafe_allow_html=True,
)
st.caption(
    "Phase 5: your question is embedded with all-MiniLM-L6-v2 and matched against "
    "the 137 chunks from the five HDFC Groww pages. Generation (Mistral) is not wired "
    "yet - below is the evidence a generator would be allowed to use."
)

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
        placeholder="e.g. What is the expense ratio of HDFC Flexi Cap?",
    )
    k = st.slider("Number of chunks (top-k)", 1, 10, DEFAULT_K)
    submitted = st.form_submit_button("Search")

question = query_text if submitted else picked

if question:
    question = question.strip()
    if not question:
        st.warning("Ask a factual question about the five HDFC funds.")
    else:
        st.session_state.last_query = question
        try:
            with st.spinner("Embedding question and searching ChromaDB..."):
                from retrieval.retriever import retrieve

                st.session_state.evidence = retrieve(question, k=k)
        except RuntimeError as exc:
            st.error(str(exc))
            st.session_state.evidence = None

st.divider()
evidence = st.session_state.get("evidence")
if evidence is None:
    st.info("Ask a factual question above to see which chunks would support an answer.")
else:
    render_evidence(evidence)

if st.button("Clear results"):
    st.session_state.evidence = None
    st.session_state.pop("ask_box", None)
    st.session_state.pop("last_query", None)
    st.rerun()