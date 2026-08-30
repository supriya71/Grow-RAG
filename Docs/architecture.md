# Architecture: Groww HDFC FAQ RAG Chatbot

**Role:** Senior architect view of the PRD prototype  
**Constraint:** This document implements **only** PRD scope. No extra sources, stores, models, or services.  
**PRD:** `Docs/PRD-Groww-HDFC-RAG-Chatbot.md`

---

## 1. System context (PRD only)

A single inspectable pipeline:

```
Allowlist of 5 Groww URLs
        │
        ▼
 Phase 1  Data loading     → page text + fetch timestamp
        │
        ▼
 Phase 2  Chunking         → chunks with fund name + URL
        │
        ▼
 Phase 3  Embedding        → vectors (all-MiniLM-L6-v2)
        │
        ▼
 Phase 4  Vector store     → local ChromaDB
        │
        ▼
 Phase 5  Retrieval logic  → top-k chunks for a question
        │
        ▼
 Phase 6  Retrieval testing → PRD corpus / citation checks

Adjacent PRD pieces (not expanded here): same embedding model on the question,
Mistral prompt (facts-only, ≤3 sentences, one citation, last-updated),
Streamlit UI, answer-policy gates (advice / returns / PII).
```

**Hard corpus (the only allowed inputs):**

| Fund | URL |
| --- | --- |
| HDFC Large Cap Fund Direct Growth | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth |
| HDFC Flexi Cap Fund Direct Growth | https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth |
| HDFC ELSS Tax Saver Fund Direct Plan Growth | https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth |
| HDFC Small Cap Fund Direct Growth | https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth |
| HDFC Balanced Advantage Fund Direct Growth | https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth |

Any URL outside this list is a defect. Flexi-cap identity is the **fund name**; the slug may still say `hdfc-equity-fund-direct-growth`.

**Prototype stack (PRD):** public HTML fetch → chunk → `sentence-transformers/all-MiniLM-L6-v2` → ChromaDB (local) → retrieve → Mistral (API key in env).

---

## Phase 1 — Data loading

**Purpose:** Pull public HTML for the five URLs and produce clean page text plus a source timestamp.

**Allowlist:** Load from a frozen list of the five URLs. No crawl, no follow-on links, no PDFs, no blogs, no app back-end.

**Per URL contract:**

| Field | Rule |
| --- | --- |
| `fund_name` | Canonical name from the PRD table (not inferred from slug alone) |
| `url` | Exact allowlisted URL |
| `text` | Visible page text after HTML parse (main content; drop chrome/nav/script noise if trivial) |
| `fetched_at` | Timestamp of this fetch — feeds `Last updated from sources:` |

**Behaviour:**

- HTTP GET public pages only.
- Simple retry on transient failure (PRD: page structure / fetch may fail).
- If a URL fails after retry: do not substitute another source; record failure; index only successful pages.
- Do not persist PAN, Aadhaar, accounts, OTPs, emails, phones (pages should not contain user PII; loader must not add any).

**Exit criteria:** Five (or fewer on failure) documents, each bound to one allowlisted URL and `fetched_at`. No sixth document.

---

## Phase 2 — Chunking

**Purpose:** Split each page so retrieval can hit a fact (expense ratio, SIP, exit load, lock-in, riskometer, benchmark) without losing which fund it belongs to.

**Per chunk contract:**

| Field | Rule |
| --- | --- |
| `chunk_id` | Stable id (e.g. fund slug + index) |
| `text` | Chunk body |
| `fund_name` | Copied from parent document |
| `url` | Copied from parent document (the single citation URL later) |
| `fetched_at` | Copied from parent document |

**Rules:**

- Chunk **per fund page**; do not merge two funds into one chunk (PRD: do not mix two funds in one uncited blob).
- Keep `fund_name` + `url` on **every** chunk (PRD 7.1).
- Size: small enough that one fact stays intact; overlapping windows are allowed if they keep headings with the numbers they describe.
- Do not inject text from outside the five pages.

**Exit criteria:** All chunks trace to exactly one of the five URLs.

---

## Phase 3 — Embedding

**Purpose:** Turn chunk text into vectors with the **only** allowed model.

**Model:** `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face). No other embedding model.

**Rules:**

- Embed **chunk `text`**. Metadata (`fund_name`, `url`, `fetched_at`) is stored, not substituted as the embedding input unless concatenated as a short prefix to help Flexi Cap vs legacy slug (optional prefix: fund name only — still PRD identity, not a new source).
- Same model **must** be used later for the user question (PRD 7.2).
- Local / free / lightweight; no paid embedding API.

**Exit criteria:** One vector per chunk, same dimensionality for all rows, same model id recorded for the collection.

---

## Phase 4 — Vector store

**Purpose:** Persist embeddings so retrieval is inspectable and local.

**Store:** ChromaDB, local persistence (PRD: local is enough). One collection for this prototype.

**Stored per item:**

- Vector  
- Document text  
- Metadata: `fund_name`, `url`, `fetched_at`

**Rules:**

- Collection contents = chunks from Phase 2 only.
- Rebuild from allowlist ingest; do not upsert arbitrary URLs.
- Persist `fetched_at` (or collection-level min/max) so the UI can print `Last updated from sources: <date>`.
- No user queries, PII, or chat logs in Chroma.

**Exit criteria:** Querying the collection returns only metadata URLs in the allowlist of five.

---

## Phase 5 — Retrieval logic

**Purpose:** Given a user question, return the chunks the generator is allowed to see.

**Flow:**

```
question
  → embed with all-MiniLM-L6-v2 (same as Phase 3)
  → similarity search in ChromaDB
  → top-k chunks + metadata
```

**Rules (PRD-aligned):**

| Situation | Retrieval behaviour |
| --- | --- |
| Named fund + fact | Prefer chunks whose `fund_name` / `url` match that fund |
| “HDFC Flexi Cap” | Must still hit the equity-fund slug page via `fund_name` metadata |
| Ambiguous “HDFC fund” | Return mixed funds; caller may ask which of the five or answer only if one fund dominates |
| Empty retrieval / very low similarity | Return empty set — generator must not invent (honest miss) |
| Question about a fund not in the five | Chunks will be off-topic or empty; do not add external search |

**What retrieval does *not* do:**

- Call Mistral (that is generation).
- Compute or compare returns.
- Follow new URLs.
- Rank “best” fund.

**Handoff to generation (PRD 7.2 / 7.3):** top-k texts + **one** citation URL (URL of the chunk(s) used; if multiple funds, do not present as a single uncited mix). `fetched_at` from those chunks → last-updated line.

**Exit criteria:** Every retrieved `url` is one of the five; Flexi Cap queries can retrieve the legacy-slug document.

---

## Phase 6 — Retrieval testing

**Purpose:** Prove the index and retriever match the PRD **before** trusting the LLM. Tests are corpus and citation tests, not “model quality” or live NAV accuracy.

**Fixtures:** Questions from PRD §9 that exercise **retrieval** (not advice/PII UI gates).

| PRD # | Query intent | Pass if retrieved chunks… |
| --- | --- | --- |
| 1 | Expense ratio / NAV / AUM / benchmark / riskometer for a **named** fund | Include that fund’s URL as top hit (or in top-k with that `fund_name`) |
| 2 | Minimum SIP / lump sum / exit load | Same fund URL as asked |
| 3 | ELSS lock-in | ELSS page URL: `.../hdfc-elss-tax-saver-fund-direct-plan-growth` |
| 4 | “HDFC Flexi Cap” (not “equity fund”) | Flexi-cap URL: `.../hdfc-equity-fund-direct-growth` |
| 5 | Ambiguous “expense ratio of HDFC fund” | Either multiple of the five URLs **or** a clear single fund — never a sixth URL |
| 8 | Fund **not** in the five | No justification to treat an off-corpus fund as cited; top hit must still be one of the five **or** empty — never a new source |
| 9 | Capital-gains statement how-to | Only chunks from the five pages; empty is a valid pass if the text is absent |
| 12 | Wrong-fund risk | Citation URL equals the retrieved fund; test fails if metadata mixes two funds in one chunk |

**Index invariants (always):**

1. Document URL set ⊆ the five allowlisted URLs.  
2. Every chunk has `fund_name` + `url` + `fetched_at`.  
3. Embeddings were produced with `sentence-transformers/all-MiniLM-L6-v2`.  
4. Chroma is local and contains no extra collections used as sources.

**Out of this phase (covered by answer policy / UI, not retriever):** PRD #6 advice refusal, #7 no return comparison, #10 PII refuse, #11 empty/gibberish. Those must not be “fixed” by adding data sources.

**Exit criteria:** Phase 6 checklist green on the table above; pipeline remains ingest → chunk → embed → Chroma → retrieve.

---

## 2. Non-goals (architecture)

Do not introduce: extra crawlers, third-party blogs, authenticated Groww APIs, non-MiniLM embeddings, non-Chroma stores, return calculators, PII stores, or production SLAs. Re-ingest cadence (button vs startup) stays an open PRD question; architecture only requires `fetched_at` to exist.
