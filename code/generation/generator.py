"""Phase 7 generation: synthesize a facts-only answer from retrieved chunks.

Calls Mistral's OpenAI-compatible chat completions endpoint over plain HTTP
(requests — already a dependency). Never invents facts outside the retrieved
chunks; if there is no key or no grounding, it falls back to a safe response.

Answer policy (PRD 7.3):
  - factual + grounded  -> <=3 sentences, one citation, last-updated
  - not in corpus       -> honest miss
  - advice / returns    -> refusal, no recommendation
  - PII                 -> refuse, do not store
  - empty / gibberish   -> short nudge
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import dotenv
import requests

# Load the local .env (MISTRAL_API_KEY) from the repo root, if present.
dotenv.load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_MODEL = "mistral-small-latest"
TIMEOUT_S = 30
SYSTEM_PROMPT = (
    "You are a facts-only assistant for a small mutual-fund FAQ over exactly five "
    "public Groww fund pages. Answer only from the provided source excerpts. "
    "Keep the answer to at most 3 short sentences and state the specific fact "
    "clearly (e.g. the exact number, years, or rule). If the excerpts do not "
    "contain the answer, say you do not have that in the indexed pages. Never "
    "invent figures, never give buy/sell/portfolio advice, and never compute or "
    "compare returns."
)

_ADVICE_RE = re.compile(
    r"\b(should i|should i (buy|sell|invest|switch|redeem)|is it (good|better|worth)"
    r"|best fund|which.*(best|better)|recommend|advice|portfolio|diversif|allocate)\b",
    re.I,
)
_RETURNS_RE = re.compile(
    r"\b(return|returns|performance|outperform|top.?performer|highest return|nav growth|past year(s)? return)\b",
    re.I,
)
_PII_RE = re.compile(
    r"\b(\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}"
    r"|[A-Z]{5}\d{4}[A-Z]|(\+?\d[\s\-()]*){10,}|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,})\b",
    re.I,
)


class ApiKeyMissingError(RuntimeError):
    """Raised when Mistral generation is requested without an API key."""

_FACT_LABELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Lock-in period", ("lock-in", "lockin", "lock in", "lock up")),
    ("Expense ratio", ("expense ratio", "expense_ratio", "mer")),
    ("Min. for SIP", ("sip", "systematic investment")),
    ("Min. for 1st investment", ("minimum investment", "lumpsum", "lump sum", "min investment")),
    ("Exit load", ("exit load", "exit_charge", "exit chrg")),
    ("NAV", ("nav", "net asset value")),
    ("Fund size (AUM)", ("aum", "assets under management", "fund size")),
    ("Rating", ("rating", "riskometer", "risk factor", "star rating")),
    ("Fund benchmark", ("benchmark", "index", "nifty")),
    ("Date of Incorporation", ("launch date", "date of incorporation", "launched", "incorporation")),
    ("Fund manager", ("fund manager", "manager")),
    ("Stamp duty", ("stamp duty",)),
    ("Tax implication", ("tax implication", "capital gains tax", "ltcg", "stcg")),
)
_VALUE_RE = re.compile(r"[%₹]|\d+(\.\d+)?(%|,| )|Nil|Very High|Moderately|Low|High|NIFTY|CRISIL", re.I)


def _topic_labels(question: str) -> tuple[str, ...]:
    q = question.lower()
    for label, keys in _FACT_LABELS:
        if any(k in q for k in keys):
            return (label,)
    return ()


def _fact_value(text: str, label: str) -> str | None:
    """Return the short value line that follows a label in a chunk's text."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for idx, line in enumerate(lines):
        if line.lower() == label.lower() or line.startswith(label):
            for nxt in lines[idx + 1: idx + 5]:
                if 0 < len(nxt) <= 70 and _VALUE_RE.search(nxt):
                    return nxt
            # e.g. "NAV: 28 Aug '26" + value on following line handled above
    return None


_RISK_RE = re.compile(r"rated\s+(Very High|High|Moderately High|Moderately Low|Low|Very Low)\s+risk", re.I)
_MANAGER_RE = re.compile(
    r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,3})\s+is the (?:Current )?Fund Manager", re.I
)


def _fallback_answer(question: str, res: dict) -> str:
    """Return the fact the question asks about, verbatim, when no LLM is possible."""
    chunks = res.get("chunks", [])
    if not chunks:
        return (
            "I don't have that information in the indexed Groww pages "
            "(no close match found)."
        )
    q = question.lower()
    text = "\n".join(c["text"] for c in chunks)
    if any(k in q for k in ("risk", "riskometer")):
        m = _RISK_RE.search(text)
        if m:
            fund = chunks[0]["fund_name"]
            return f"{fund}: Risk — {m.group(1).title()} risk"
    if "fund manager" in q:
        m = _MANAGER_RE.search(text)
        if m:
            fund = chunks[0]["fund_name"]
            return f"{fund}: Fund manager — {m.group(1)}"
    labels = _topic_labels(question)
    for label in labels:
        for chunk in chunks:
            value = _fact_value(chunk["text"], label)
            if value:
                return f"{chunk['fund_name']}: {label} — {value}"
    # No explicit topic (or topic value not found): return a relevant short pair.
    for label in ("Lock-in period", "Expense ratio", "Exit load", "Min. for SIP",
                  "Fund size (AUM)", "NAV", "Rating"):
        for chunk in chunks:
            value = _fact_value(chunk["text"], label)
            if value:
                return f"{chunk['fund_name']}: {label} — {value}"
    top = chunks[0]
    compact = " ".join(top["text"].split())
    return f"{top['fund_name']}: {compact[:240]}"


def answer_policy_type(question: str) -> str:
    """Classify the question into a PRD 7.3 gate: refuse/nudge/answer."""
    if not (question or "").strip():
        return "empty"
    if _PII_RE.search(question):
        return "pii"
    if _ADVICE_RE.search(question):
        return "advice"
    if _RETURNS_RE.search(question):
        return "returns"
    return "answer"


def generate(question: str, res: dict) -> dict:
    """Run answer policy + (optional) Mistral generation on retrieval evidence.

    Returns:
        {
          "query", "answer", "citation_url", "citation_label",
          "last_updated", "policy", "empty"
        }
    """
    question = (question or "").strip()
    policy = answer_policy_type(question)
    fetched = res.get("fetched_at_max")

    # Policy gates that never need the LLM.
    if policy == "empty":
        return {
            "query": question,
            "policy": "empty",
            "answer": "Please ask a factual question about these five HDFC funds.",
            "citation_url": None,
            "citation_label": None,
            "last_updated": fetched,
            "empty": True,
        }
    if policy == "pii":
        return {
            "query": question,
            "policy": "pii",
            "answer": (
                "I can't process personal identifiers (PAN, account, phone, email). "
                "I don't collect or store any personal information."
            ),
            "citation_url": None,
            "citation_label": None,
            "last_updated": fetched,
            "empty": True,
        }
    if policy in ("advice", "returns"):
        citation_url = _single_citation(res)
        label = _citation_label(citation_url)
        return {
            "query": question,
            "policy": policy,
            "answer": (
                "I only answer factual questions from the five fund pages and "
                "don't give buy/sell/portfolio advice or compare returns. See the "
                "fund's Groww page for the official factsheet and figures: "
                f"{citation_url}."
            ),
            "citation_url": citation_url,
            "citation_label": label,
            "last_updated": fetched,
            "empty": False,
        }

    # Factual path: grounded answer from retrieved chunks.
    if res.get("empty") or not res.get("chunks"):
        return {
            "query": question,
            "policy": "answer",
            "answer": (
                "I don't have that in the indexed Groww pages — no close match "
                "was found, and I won't invent it."
            ),
            "citation_url": None,
            "citation_label": None,
            "last_updated": fetched,
            "empty": True,
        }

    citation_url = _single_citation(res)
    label = _citation_label(citation_url)
    answer = _synthesize(question, res)
    return {
        "query": question,
        "policy": "answer",
        "answer": answer,
        "citation_url": citation_url,
        "citation_label": label,
        "last_updated": fetched,
        "empty": False,
    }


def _single_citation(res: dict) -> str | None:
    urls = res.get("citation_urls") or []
    return urls[0] if urls else None


_CORPUS_FUND = {
    "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth": "HDFC Large Cap",
    "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth": "HDFC Flexi Cap",
    "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth": "HDFC ELSS Tax Saver",
    "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth": "HDFC Small Cap",
    "https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth": "HDFC Balanced Advantage",
}


def _citation_label(url: str | None) -> str | None:
    if not url:
        return None
    return _CORPUS_FUND.get(url, "Groww fund page")


def _synthesize(question: str, res: dict) -> str:
    key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if not key:
        return _fallback_answer(question, res)

    excerpts = []
    for i, chunk in enumerate(res.get("chunks", [])[:5], 1):
        excerpts.append(f"[{i}] {chunk['fund_name']}\n{chunk['text']}")
    context = "\n\n".join(excerpts)
    user_prompt = (
        f"Source excerpts (each tagged with its fund name):\n{context}\n\n"
        f"Question: {question}\n\n"
        "Give the factual answer in at most 3 sentences using only the excerpts. "
        "If the excerpts don't answer it, say so."
    )

    try:
        resp = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": DEFAULT_MODEL,
                "temperature": 0.0,
                "max_tokens": 200,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=TIMEOUT_S,
        )
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"].strip()
        return answer or _fallback_answer(question, res)
    except (requests.RequestException, KeyError, ValueError):
        return _fallback_answer(question, res)
