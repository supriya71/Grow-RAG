# PRD: Groww HDFC Mutual Fund FAQ Chatbot (RAG Prototype)

**Product:** Facts-only RAG chatbot for five HDFC Direct Growth funds on Groww  
**Type:** Working prototype / hobby project to test RAG  
**Source of truth:** Public Groww fund pages only  
**Owner:** PM (prototype)  
**Status:** Draft  
**Last updated:** 30 Aug 2026

---

## 1. Problem

Users looking at HDFC funds on Groww still ask the same factual questions: expense ratio, SIP minimum, exit load, lock-in, riskometer, benchmark, and how to get a capital-gains statement. Today that means hunting across fund pages. Opinionated “should I buy/sell?” questions should not be answered as advice.

This prototype proves a small, source-grounded RAG loop can answer those facts with a citation, without becoming an advisor.

## 2. Goal

Ship a Streamlit FAQ assistant that:

1. Retrieves from **only** five specified Groww URLs.
2. Answers **factual** questions in **≤3 sentences**, with **one citation link** and a **last-updated-from-sources** line.
3. **Refuses** investment advice / portfolio opinions politely, with a relevant educational link.
4. Uses a lightweight, free embedding model and ChromaDB so the RAG pipeline is easy to inspect.

**Non-goal:** A production Groww product, multi-AMC coverage, return calculations, or personalised recommendations.

## 3. In scope

| Area | Decision |
| --- | --- |
| Website | [groww.in](https://groww.in/) |
| AMC | HDFC only |
| Corpus | Exactly the five fund pages below — nothing else |
| UX | Tiny Streamlit UI, Groww-inspired colours |
| LLM | Mistral API |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | ChromaDB |

### Corpus (hard boundary)

| Category | Fund | URL |
| --- | --- | --- |
| Large-cap | HDFC Large Cap Fund Direct Growth | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth |
| Flexi-cap | HDFC Flexi Cap Fund Direct Growth | https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth |
| ELSS | HDFC ELSS Tax Saver Fund Direct Plan Growth | https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth |
| Small-cap | HDFC Small Cap Fund Direct Growth | https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth |
| Hybrid | HDFC Balanced Advantage Fund Direct Growth | https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth |

Note: Flexi-cap still uses Groww’s legacy `hdfc-equity-fund-direct-growth` slug.

---

## 4. Out of scope

- Any URL, PDF, blog, or screenshot that is not one of the five pages
- App back-end / authenticated Groww data
- PII: PAN, Aadhaar, account numbers, OTPs, emails, phone numbers (do not accept or store)
- Computing or comparing returns / performance claims (link to official factsheet on the fund page if asked)
- Multi-turn portfolio construction, KYC, transactions, or login
- Heavy / paid embedding models

---

## 5. Users and jobs-to-be-done

**Primary user:** Someone comparing these five HDFC Direct Growth schemes who wants a number or rule, not advice.

| Job | Example |
| --- | --- |
| Lookup a fact on a named fund | “Expense ratio of HDFC Large Cap Direct Growth?” |
| Category rule | “ELSS lock-in?” |
| Operational how-to (if present on page) | “How to download capital-gains statement?” |
| Know when not to get advice | “Should I sell my small-cap?” → refusal |

---

## 6. User experience

**Layout:** Single Streamlit page.

**Must show:**

- Welcome line
- Three example questions (clickable or copy-paste)
- Persistent note: **“Facts-only. No investment advice.”**
- Chat input + answer area
- Every answer: one citation URL + `Last updated from sources: <date>`

**Visual:** Groww-like palette (dark green / mint / white / dark text — match groww.in, not a pixel-perfect clone).

**Suggested example questions:**

1. What is the expense ratio of HDFC Large Cap Fund Direct Growth?
2. What is the lock-in for HDFC ELSS Tax Saver?
3. What is the minimum SIP and exit load for HDFC Small Cap Fund Direct Growth?

---

## 7. Functional requirements

### 7.1 Ingestion

- Fetch/parse **only** the five URLs (public HTML).
- Chunk page text for retrieval (keep fund identity on each chunk: fund name + URL).
- Embed with `sentence-transformers/all-MiniLM-L6-v2`.
- Persist in ChromaDB (local is enough for the prototype).
- Record source fetch timestamp for the “Last updated from sources” line.

### 7.2 Retrieval and generation

- Embed the user question with the **same** model.
- Retrieve the most relevant chunks from ChromaDB.
- Build a constrained prompt for **Mistral**: answer only from retrieved chunks; if missing, say so.
- Generate the user-facing answer.

### 7.3 Answer policy

| Query type | Behaviour |
| --- | --- |
| Factual, in corpus | ≤3 sentences; one citation link (the Groww fund URL used); last-updated line |
| Factual, not in the five pages | “I don’t have that in the indexed Groww pages” + do not invent |
| Advice / buy-sell / “what should I do” | Polite refusal; facts-only reminder; one relevant educational link (from the same five pages if possible, e.g. riskometer / scheme overview) |
| Performance / returns / comparison of returns | Do not compute or compare; point to factsheet / official figures on the cited Groww page |
| PII in the query | Refuse to process identifiers; remind that the bot does not collect PII |

---

## 8. RAG pipeline (prototype)

```
[5 Groww URLs]
    → scrape/parse
    → chunk
    → embed (all-MiniLM-L6-v2)
    → ChromaDB

[User question]
    → embed (same model)
    → retrieve top-k chunks
    → LLM prompt (Mistral)
    → answer + 1 citation + last-updated
```

**Config (prototype):** Mistral API key via environment variable (never commit keys).

---

## 9. Edge cases to test

| # | Case | Expected |
| --- | --- | --- |
| 1 | Expense ratio / NAV / AUM / benchmark / riskometer for a named fund | Fact + that fund’s URL |
| 2 | Minimum SIP / min lump sum / exit load | Fact + citation |
| 3 | ELSS lock-in (3 years) | Fact + ELSS page URL |
| 4 | Flexi-cap asked as “HDFC Flexi Cap” but page slug is equity fund | Still retrieves the flexi-cap page |
| 5 | Ambiguous fund (“expense ratio of HDFC fund”) | Ask which of the five, or answer only if retrieval is unambiguous |
| 6 | “Should I buy/sell/switch?” | Refusal + educational link, no recommendation |
| 7 | “Which of these is best / highest return?” | No ranking or computed returns; factsheet / refuse comparison |
| 8 | Question about a fund not in the five | Out-of-corpus refusal |
| 9 | How to download capital-gains statement | Only if present on indexed pages; else honest miss |
| 10 | User pastes PAN / phone / email | Do not store; refuse PII |
| 11 | Empty / gibberish input | Short “ask a factual question about these five funds” |
| 12 | Retrieval mismatch (wrong fund) | Prefer citing the retrieved URL; do not mix two funds in one uncited blob |

---

## 10. Success criteria (prototype)

- Corpus is strictly the five URLs (no extra sources in the index).
- Happy-path factual questions return a grounded answer, **one** Groww citation, ≤3 sentences, and last-updated line.
- Advice questions never produce a buy/sell recommendation.
- UI shows welcome, 3 examples, and the facts-only disclaimer.
- Pipeline is inspectable: ingest → chunk → embed → Chroma → retrieve → Mistral.

**Explicitly not success:** Accuracy of live Groww numbers beyond what was scraped at ingest time; production latency/SLA; SEBI-compliant advisor behaviour beyond the facts-only refusal.

---

## 11. Risks and constraints

- **Stale data:** Answers reflect last scrape, not live Groww. Always show last-updated.
- **Page structure:** Groww HTML may change; ingestion may need a simple retry/parse fallback.
- **Hallucination:** Prompt must forbid answering from model memory; empty retrieval → no invented ratios.
- **Compliance:** This is a hobby FAQ over public pages, not investment advice. Disclaimer is mandatory.
- **Secrets:** Mistral key only in env / local secrets.

---

## 12. Build sequence (suggested)

1. Ingest five URLs → chunks → ChromaDB.  
2. CLI retrieve + Mistral answer (no UI).  
3. Streamlit shell (welcome, examples, disclaimer, colours).  
4. Wire chat + citation + last-updated.  
5. Edge-case test list above.  

---

## 13. Open questions (can ship prototype without resolving)

- Re-ingest cadence (manual button vs on app start).  
- Exact Groww hex colours vs close approximation.  
- Whether “capital-gains statement” exists as text on these five pages; if not, honest miss is correct.
