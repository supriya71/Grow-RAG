"""Split one fund page at a time. Never merge two funds into one chunk."""

from __future__ import annotations

import re

# MiniLM-friendly window: keep a heading with the number on the next line.
CHUNK_SIZE_CHARS = 800
OVERLAP_CHARS = 180

_VALUE_RE = re.compile(r"[\d%₹]|Very High|Moderately|NIFTY|BSE|CRISIL", re.I)
_CONTENT_STARTS = ("NAV:", "Min. for SIP", "Expense ratio", "Fund size")
_FOOTER_STARTS = ("Download the App", "© 2016", "Vaishnavi Tech Park")


def _is_label(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return False
    if _VALUE_RE.search(stripped) and any(ch.isdigit() for ch in stripped):
        return False
    return True


def _is_value(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and len(stripped) <= 80 and bool(_VALUE_RE.search(stripped))


def label_value_units(text: str) -> list[str]:
    """Keep short labels glued to the following value line (expense ratio → 1.02%)."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    units: list[str] = []
    i = 0
    while i < len(lines):
        if i + 1 < len(lines) and _is_label(lines[i]) and _is_value(lines[i + 1]):
            units.append(f"{lines[i]}\n{lines[i + 1]}")
            i += 2
            continue
        units.append(lines[i])
        i += 1
    return units


def content_window(text: str) -> str:
    """Drop leading/trailing site chrome; still only text from this page."""
    starts = [text.find(marker) for marker in _CONTENT_STARTS]
    found = [idx for idx in starts if idx != -1]
    start = min(found) if found else 0

    end = len(text)
    for marker in _FOOTER_STARTS:
        idx = text.find(marker)
        if idx != -1:
            end = min(end, idx)
    if end <= start:
        end = len(text)
    window = text[start:end].strip()
    return window or text.strip()


def overlapping_windows(units: list[str], size: int, overlap: int) -> list[str]:
    if not units:
        return []

    def joined_len(parts: list[str]) -> int:
        if not parts:
            return 0
        return sum(len(p) for p in parts) + (len(parts) - 1)

    chunks: list[str] = []
    start = 0
    n = len(units)

    while start < n:
        buf: list[str] = []
        i = start
        while i < n:
            extra = len(units[i]) + (1 if buf else 0)
            if buf and joined_len(buf) + extra > size:
                break
            buf.append(units[i])
            i += 1
            if joined_len(buf) >= size:
                break

        if not buf:
            chunks.append(units[start])
            start += 1
            continue

        chunks.append("\n".join(buf))
        if i >= n:
            break

        overlap_len = 0
        new_start = i
        for offset, unit in enumerate(reversed(buf)):
            overlap_len += len(unit) + (1 if overlap_len else 0)
            new_start = i - 1 - offset
            if overlap_len >= overlap:
                break
        start = new_start if new_start > start else start + 1

    return chunks


def chunk_page_text(text: str) -> list[str]:
    units = label_value_units(content_window(text))
    return overlapping_windows(units, CHUNK_SIZE_CHARS, OVERLAP_CHARS)
