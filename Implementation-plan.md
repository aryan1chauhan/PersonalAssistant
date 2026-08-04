# Implementation Plan: SecondSelf — Personal AI Second Brain

This implementation plan outlines the step-by-step phases required to construct SecondSelf based on [architecture.md](file:///d:/PersonalAssistant/architecture.md) and [ProblemStatement.md](file:///d:/PersonalAssistant/ProblemStatement.md).

---

## Phase 0: Setup & Repository Scaffolding

### Objective
Establish the foundational directory layout, environment configuration, dependency management, and base tracking files.

### Tasks
- [x] Create repository directory tree:
  - `raw/` and `raw/assets/`
  - `wiki/1_Projects/`, `wiki/2_Areas/`, `wiki/3_Resources/`, `wiki/4_Archives/`
  - `src/` module directory
  - `docs/` documentation directory
  - `static/` asset directory
- [x] Create `requirements.txt` with required dependencies:
  - CLI & Utilities: `pydantic`, `python-dotenv`, `click`, `rich`
  - Parsing & Scraping: `requests`, `beautifulsoup4`, `trafilatura`, `pdfplumber`, `pypdf`
  - Embeddings & AI: `sentence-transformers`, `torch`, `groq`, `google-genai`, `openai`
  - App UI & Visualization: `streamlit`, `pyvis`, `networkx`
  - Testing: `pytest`, `tenacity`
- [x] Create `.env.example` template for API keys (`GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`).
- [x] Create `.gitignore` ignoring `.env`, `__pycache__`, `venv/`, and `.chroma/`.
- [x] Create base `README.md` and package initializer `src/__init__.py`.

---

## Phase 1: Week 1 — Capture Pipeline ("The Archivist")

### Objective
Build a unified CLI ingestion engine (`src/capture.py`) that captures text notes, web URLs, and local files into an append-only `raw/` folder with guaranteed timestamps and unique IDs.

### Tasks
- [x] Implement utility helper routines (`src/utils.py`):
  - ISO-8601 timestamp generator.
  - Unique ID generator format (`raw_YYYYMMDD_HHMMSS_{UUID8}`).
  - SHA-256 asset checksum calculator.
- [x] Implement text note ingestion handler.
- [x] Implement web URL scraper (`requests` + `trafilatura` / `beautifulsoup4` to extract page title, author, and clean Markdown text).
- [x] Implement file parser (`pdfplumber` for PDF text extraction, saving raw attachment files into `raw/assets/`).
- [x] Construct unified CLI entrypoint (`src/capture.py`):
  - Command: `python src/capture.py note --content "..."`
  - Command: `python src/capture.py link --url "..."`
  - Command: `python src/capture.py file --path "..."`
- [x] Test ingestion on 10+ real scattered items (notes, URLs, PDFs).

### Verification
- Acceptance Criteria:
  - [x] `raw/` and `wiki/` directory structure exists.
  - [x] One command captures a note, a link, AND a file.
  - [x] Every captured item has a timestamp + unique ID.
  - [x] 10+ real items captured in `raw/`.
- 🏅 **Badge**: The Archivist

---

## Phase 2: Week 2.1 — AI Classification Engine ("The Sorting Hat")

### Objective
Transform raw captures into organized Markdown notes within `wiki/` using LLM PARA categorization.

### Tasks
- [ ] Create LLM classifier module (`src/classify.py`).
- [ ] Formulate structured prompt targeting Groq / Llama 3 / Gemini:
  - Categorizes note into `1_Projects`, `2_Areas`, `3_Resources`, or `4_Archives`.
  - Extracts 3-5 normalized tags.
  - Generates a concise one-line summary and clean note title.
- [ ] Read captured items from `raw/`, send payloads to LLM, and write structured Markdown files with YAML frontmatter to `wiki/{category}/{slug}.md`.

---

## Phase 3: Week 2.2 — Dense Embeddings & Vector Auto-Linking ("Connect the Dots")

### Objective
Compute local vector embeddings per note and automatically inject bidirectional `[[wikilinks]]` between related notes without manual tagging.

### Tasks
- [ ] Create embedding & auto-linking module (`src/link.py`).
- [ ] Load `sentence-transformers/all-MiniLM-L6-v2` embedding model.
- [ ] Compute 384-dimensional dense vectors for all Markdown notes in `wiki/`.
- [ ] Perform pairwise Cosine Similarity calculation across notes.
- [ ] For pairs with similarity score $\ge 0.65$, automatically append bidirectional `[[Related Note Title]]` links into each note's Markdown file under `## Related Knowledge`.
- [ ] Run pipeline across 15+ real items.

### Verification
- Acceptance Criteria:
  - [ ] Any raw capture → category + tags + summary automatically.
  - [ ] PARA categorization working.
  - [ ] Embeddings computed per note.
  - [ ] Related notes auto-linked (no manual tagging).
  - [ ] Runs on 15+ real items → organized `wiki/`.
- 🏅 **Badge**: The Librarian

---

## Phase 4: Week 3 — Knowledge Graph Engine ("The Cartographer")

### Objective
Convert `wiki/` notes into a node-edge graph data model exportable as `graph.json`, rendered as an interactive force-directed visual brain.

### Tasks
- [ ] **Phase 4.1: Graph Data Model Exporter (`src/build_graph.py`)**:
  - Parse all Markdown files in `wiki/`.
  - Extract nodes (notes, categories, tags) and edges (explicit wikilinks + vector similarity edges).
  - Export structured `graph.json`.
- [ ] **Phase 4.2: Interactive Graph Renderer (vis-network / PyVis)**:
  - Create HTML template (`static/graph_template.html`).
  - Render notes as color-coded pulsing nodes matching PARA categories.
  - Implement hover popups displaying note summary & content preview, zoom, pan, and drag physics.

### Verification
- Acceptance Criteria:
  - [ ] Script builds nodes + edges from notes and exports clean `graph.json`.
  - [ ] Interactive force-directed graph renders from `graph.json`.
  - [ ] Hover reveals note content popup.
  - [ ] Drag + zoom work smoothly.
  - [ ] Built from real notes, not dummy data.
- 🏅 **Badge**: The Cartographer

---

## Phase 5: Week 4 — RAG Engine & Streamlit Application ("The Oracle")

### Objective
Wire up natural language retrieval-augmented search over personal notes and package the full system into a single deployable Streamlit dashboard.

### Tasks
- [ ] **Phase 5.1: Retrieval-Augmented Q&A Engine (`src/ask.py`)**:
  - Embed user natural language question.
  - Perform vector search across notes to retrieve top $k=3..5$ matching context passages.
  - Pass context + question to LLM synthesizer with instructions to provide grounded answers with source file citations.
- [ ] **Phase 5.2: Streamlit Dashboard (`app.py`)**:
  - Tab 1: **Living Brain Visualizer** (embedded vis-network graph).
  - Tab 2: **Ask Your Brain** (search bar, streaming response, citation drawer).
  - Tab 3: **Quick Capture** (text note, URL link, file upload interface).

### Verification
- Acceptance Criteria:
  - [ ] `ask()` returns answers synthesized from your own notes (retrieval + LLM).
  - [ ] One Streamlit app contains both the graph and the search bar.
  - [ ] Deployed live with a public URL.
  - [ ] Full pipeline works end-to-end in the deployed app.
- 🏅 **Badge**: The Oracle

---

## Phase 6-7: Local Testing, Validation & Edge Case Handling

### Tasks
- [ ] Write integration test suite (`pytest`) verifying ingestion, classification, linking, graph generation, and RAG retrieval.
- [ ] Test edge cases documented in `edge-case.md` (unreachable URLs, scanned PDFs, rate limits, short notes, missing API keys).
- [ ] Verify full pipeline workflow locally: Ingest → Classify → Link → Build Graph → Ask Query.

---

## Phase 8-9: Public Deployment & Final Round of Verification

### Tasks
- [ ] Configure repository for cloud deployment (Streamlit Community Cloud / HuggingFace Spaces / Render).
- [ ] Add cloud environment secrets (`GROQ_API_KEY`, `GEMINI_API_KEY`).
- [ ] Deploy application and get public live URL.
- [ ] Perform final end-to-end user verification on the live public URL.
- [ ] Finalize README setup instructions and push repository to GitHub.
