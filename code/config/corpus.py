"""Frozen allowlist: exactly the five PRD / architecture URLs. Nothing else."""

from typing import TypedDict


class CorpusEntry(TypedDict):
    fund_name: str
    url: str
    slug: str


CORPUS: tuple[CorpusEntry, ...] = (
    {
        "fund_name": "HDFC Large Cap Fund Direct Growth",
        "url": "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
        "slug": "hdfc-large-cap-fund-direct-growth",
    },
    {
        "fund_name": "HDFC Flexi Cap Fund Direct Growth",
        "url": "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
        "slug": "hdfc-equity-fund-direct-growth",
    },
    {
        "fund_name": "HDFC ELSS Tax Saver Fund Direct Plan Growth",
        "url": "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
        "slug": "hdfc-elss-tax-saver-fund-direct-plan-growth",
    },
    {
        "fund_name": "HDFC Small Cap Fund Direct Growth",
        "url": "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
        "slug": "hdfc-small-cap-fund-direct-growth",
    },
    {
        "fund_name": "HDFC Balanced Advantage Fund Direct Growth",
        "url": "https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth",
        "slug": "hdfc-balanced-advantage-fund-direct-growth",
    },
)

ALLOWLIST_URLS: frozenset[str] = frozenset(entry["url"] for entry in CORPUS)
ALLOWLIST_SIZE = 5


def validate_corpus() -> None:
    if len(CORPUS) != ALLOWLIST_SIZE:
        raise RuntimeError(f"Allowlist must contain exactly {ALLOWLIST_SIZE} entries")
    urls = [entry["url"] for entry in CORPUS]
    if len(set(urls)) != ALLOWLIST_SIZE:
        raise RuntimeError("Allowlist URLs must be unique")
    if not ALLOWLIST_URLS.issubset(set(urls)):
        raise RuntimeError("Allowlist inconsistency")
