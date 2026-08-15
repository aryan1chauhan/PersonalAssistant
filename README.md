# 🧠 SecondSelf — Your Personal AI Second Brain

> Capture anything → AI organizes it → Knowledge auto-links → Visual brain graph → Ask questions & get answers from your own notes.

---

## What It Does

**SecondSelf** turns scattered notes, bookmarks, and files into an interconnected, searchable personal knowledge base — powered by AI.

| Feature | Description |
|---|---|
| **📥 Smart Capture** | Ingest text notes, web URLs, or file uploads from a single interface. |
| **🏷️ AI Classification** | Automatically categorizes everything into the PARA system (Projects, Areas, Resources, Archives) using LLMs. |
| **🔗 Semantic Auto-Linking** | Dense vector embeddings find related notes and inject bidirectional `[[wikilinks]]` — no manual tagging needed. |
| **🗺️ Interactive Knowledge Graph** | A force-directed vis-network graph visualizes your entire brain with hover previews, drag, and zoom. |
| **🔮 Ask Your Brain (RAG)** | Ask natural language questions and get answers grounded in your personal notes, with source citations. |

---

## Tech Stack

- **Frontend**: Streamlit dashboard with dark-mode UI
- **AI Classification**: Groq (Llama 3) → Gemini → OpenAI fallback chain
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim dense vectors)
- **Graph Engine**: NetworkX + vis-network.js
- **RAG**: Smart-chunk retrieval with cosine similarity + LLM answer synthesis

---

## Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/aryan1chauhan/PersonalAssistant.git
cd PersonalAssistant

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API Keys
```bash
cp .env.example .env
```
Edit `.env` and add at least one LLM provider key:
```
GROQ_API_KEY=gsk_your_key_here
GEMINI_API_KEY=AIzaSy_your_key_here
OPENAI_API_KEY=sk-proj_your_key_here
```

### 3. Run
```bash
streamlit run app.py
```

---

## Live Demo

Deployed on Streamlit Community Cloud — [Launch App →](https://secondself-brain.streamlit.app)

---

## License

MIT
