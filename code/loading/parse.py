"""Visible text from a single allowlisted HTML page. Does not follow links."""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

_NOISE_TAGS = ("script", "style", "noscript", "svg", "iframe", "canvas", "template")
_CHROME_TAGS = ("nav", "footer", "header", "aside")
_WS_RE = re.compile(r"[ \t]+")
_BLANK_RE = re.compile(r"\n{3,}")


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
    return _BLANK_RE.sub("\n\n", text).strip()
