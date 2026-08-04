# Edge Cases & Corner Scenarios: SecondSelf — Personal AI Second Brain

This document catalogues failure modes, edge cases, corner scenarios, and mitigation strategies across all system modules based on [architecture.md](file:///d:/PersonalAssistant/architecture.md) and [Implementation-plan.md](file:///d:/PersonalAssistant/Implementation-plan.md).

---

## 1. Ingestion & Capture Engine (`src/capture.py`)

| Corner Scenario / Edge Case | Failure Mode / Impact | Technical Mitigation Strategy |
| :--- | :--- | :--- |
| **Invalid or Unreachable Web URL** (HTTP 404, 500, SSL Error, DNS Failure) | Scraper crash or unhandled request exception. | Wrap network calls in `requests.RequestException` try/catch block. Save raw URL as string in `raw/{id}.json` with metadata `fetch_status: "failed"`. |
| **Paywalled or Javascript-Heavy Web Pages** (Medium, Substack, SPA Apps) | Empty text payload extracted by standard BeautifulSoup parser. | Use `trafilatura` primary extraction engine with fallback to `<meta name="description">` header tags and user prompt for manual text snippet fallback. |
| **Scanned or Non-Text PDF Files** | Zero text extracted by standard PDF text parser (`pdfplumber`). | Detect empty string output post-parsing. Mark metadata `requires_ocr: true`, save original asset to `raw/assets/`, and issue non-blocking user warning. |
| **Duplicate Capture Submissions** | Redundant JSON creation and duplicated wiki notes. | Compute SHA-256 hash of normalized raw content string or target URL. If hash matches existing capture, log duplicate notice and return existing capture ID. |
| **Empty or Whitespace-Only Text Notes** | Creation of useless blank raw records. | Validate string length before saving ($n > 0$ non-whitespace chars). Reject empty input with error message (`"Note content cannot be empty."`). |
| **Extremely Large Files (>25 MB)** | High memory consumption or process slowdown. | Limit file payload attachment size to 25 MB and truncate text payload preview to first 50,000 characters. |

---

## 2. AI Classification & Structuring (`src/classify.py`)

| Corner Scenario / Edge Case | Failure Mode / Impact | Technical Mitigation Strategy |
| :--- | :--- | :--- |
| **LLM Rate Limit (HTTP 429) or Quota Exhaustion** | Ingestion pipeline stalls during PARA classification. | Wrap API calls with `tenacity` retry decorator featuring exponential backoff. Provide fallback rule-based classifier for offline usage. |
| **Malformed JSON Returned by LLM** | Ingestion parser crash when mapping LLM output. | Enforce JSON output mode where supported; use regex fallback to extract `{...}` JSON blocks; retry up to 3 times on parse failure. |
| **Ambiguous PARA Category Selection** | Note placed in incorrect PARA folder. | Supply LLM prompt with strict priority hierarchy: `Projects` (active goal/deadline) > `Areas` (ongoing duty) > `Resources` (reference topic) > `Archives` (inactive). |
| **Non-English / Multilingual Captures** | Mismatched tags or inconsistent categorization. | Instruct LLM prompt to detect language, generate tags in English, and preserve original text in note content. |

---

## 3. Embedding Calculation & Auto-Linking (`src/link.py`)

| Corner Scenario / Edge Case | Failure Mode / Impact | Technical Mitigation Strategy |
| :--- | :--- | :--- |
| **Very Short Notes (<15 words)** | Noisy dense vector representations producing false link matches. | Skip auto-linking vector search for notes under 15 words. |
| **Hyper-Dense Over-Linking** | Visual clutter with dozens of weakly related notes. | Enforce strict Cosine Similarity threshold ($\ge 0.65$) and cap maximum auto-inserted links per note to top 5 most similar items. |
| **Self-Referential Links & Duplicate Links** | Note linking to itself or repeating existing links. | Exclude self-matching slug pairs and parse existing `[[wikilinks]]` prior to appending new link blocks. |

---

## 4. Knowledge Graph Engine & Visualization (`src/build_graph.py` & Graph UI)

| Corner Scenario / Edge Case | Failure Mode / Impact | Technical Mitigation Strategy |
| :--- | :--- | :--- |
| **Isolated Singleton Nodes (No links)** | Fragmented or sparse visual layout. | Render singleton nodes with smaller node radius and allow UI toggle to filter unlinked nodes in visual graph. |
| **Large Note Scale (>200 nodes)** | Browser DOM slowdown during force-directed layout simulation. | Optimize physics engine settings using `barnesHut` solver and disable continuous physics iteration once stabilized. |
| **Special Characters / Quotes in Note Titles** | JavaScript execution or HTML template injection crash. | Escape quotes, backslashes, and HTML entities in note summaries and titles prior to `graph.json` generation. |

---

## 5. Oracle RAG Engine & Q&A (`src/ask.py`)

| Corner Scenario / Edge Case | Failure Mode / Impact | Technical Mitigation Strategy |
| :--- | :--- | :--- |
| **Questions Unrelated to Ingested Notes** | LLM hallucinating information from pre-trained memory. | Enforce strict system prompt: *"If the provided context notes do not contain sufficient evidence, state clearly: 'I could not find an answer in your personal notes.'"* |
| **Contradictory Notes Written at Different Times** | Conflicting synthesized answers. | Pass ISO timestamps in retrieved note context so LLM synthesizer can highlight information evolution over time. |
| **Context Window Overflow** | Prompt token limit exceeded error. | Truncate passages to top $k=3..5$ chunks matching query vector score and cap total prompt context to 3,500 tokens. |

---

## 6. Public Deployment & Cloud Environment (`app.py`)

| Corner Scenario / Edge Case | Failure Mode / Impact | Technical Mitigation Strategy |
| :--- | :--- | :--- |
| **Missing Cloud Secret API Keys** | Application crash on public launch. | Check `os.environ` on application startup and display setup guidance screen if API keys are missing. |
| **Read-Only Serverless Filesystem** | Failure when capturing new items in deployed app. | Route temporary runtime file modifications to `/tmp` directory or session state memory when in serverless mode. |
