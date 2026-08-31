"""Visible text from a single allowlisted HTML page. Does not follow links.

Structured facts that Groww renders only inside the __NEXT_DATA__ JSON blob
(e.g. the ELSS lock-in period) are not visible in the DOM, so the plain text
parser can never see them. We extract those fields explicitly and append them
to the visible text so downstream chunking/embedding can answer them.
"""

from __future__ import annotations

import json
import re

from bs4 import BeautifulSoup

_NOISE_TAGS = ("script", "style", "noscript", "svg", "iframe", "canvas", "template")
_CHROME_TAGS = ("nav", "footer", "header", "aside")
_NEXT_DATA_RE = re.compile(
    r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>',
    re.S | re.I,
)
_WS_RE = re.compile(r"[ \t]+")
_BLANK_RE = re.compile(r"\n{3,}")


def _structured_facts_lines(html: str) -> list[str]:
    """Label/value fact lines recovered from the embedded __NEXT_DATA__ JSON.

    Returns lines like ["Lock-in period", "3 years"]. These fields are only
    present in the JSON blob, not the rendered DOM, so the standard parser
    would otherwise lose them.
    """
    match = _NEXT_DATA_RE.search(html)
    if not match:
        return []
    try:
        payload = json.loads(match.group(1))
        data = payload["props"]["pageProps"]["mfServerSideData"]
    except (ValueError, KeyError, TypeError):
        return []

    facts: list[str] = []
    lock_in = (data or {}).get("lock_in")
    if isinstance(lock_in, dict):
        years = lock_in.get("years") or 0
        months = lock_in.get("months") or 0
        days = lock_in.get("days") or 0
        if years or months or days:
            parts = []
            if years:
                parts.append(f"{years} year{'s' if years != 1 else ''}")
            if months:
                parts.append(f"{months} month{'s' if months != 1 else ''}")
            if days:
                parts.append(f"{days} day{'s' if days != 1 else ''}")
            facts.append("Lock-in period")
            facts.append(", ".join(parts))
    return facts


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    for tag in soup(_CHROME_TAGS):
        tag.decompose()

    root = soup.find("main") or soup.body or soup
    text = root.get_text(separator="\n", strip=True)
    lines = [_WS_RE.sub(" ", line).strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    text = _BLANK_RE.sub("\n\n", text).strip()

    structured = _structured_facts_lines(html)
    if structured:
        facts_block = "\n".join(structured)
        # Insert the facts beside the exit-load / tax section where related
        # facts already live, so the fact forms a clean, in-context chunk.
        insert_at = text.find("Tax implication")
        if insert_at == -1:
            insert_at = len(text)
        head = text[:insert_at].rstrip()
        tail = text[insert_at:].lstrip()
        text = f"{head}\n\n{facts_block}\n\n{tail}".strip()
    return text
