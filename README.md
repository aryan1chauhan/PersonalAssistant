# SecondSelf — Your Personal AI Second Brain

> Capture anything (note, link, file) → AI classifies & files it (PARA) → AI auto-links related knowledge → Renders a living interactive graph → Ask questions in plain English & get answers from your personal knowledge base.

---

## 🌟 System Overview

**SecondSelf** solves the fundamental flaw of traditional notes apps: information goes in, but nothing comes back out. SecondSelf automatically organizes, links, visualizes, and answers questions using your accumulated personal notes.

### End-to-End Pipeline
```
Capture any note/link/file (Week 1)
  ↓
AI classifies & files it via PARA Method (Week 2)
  ↓
AI auto-links related notes using dense embeddings (Week 2)
  ↓
Everything renders as an interactive hoverable visual brain graph (Week 3)
  ↓
Ask plain-English questions → Grounded answers synthesized from YOUR notes (Week 4)
  ↓
Deployed on a public cloud URL (Week 4)
```

---

## 📁 Repository Structure

```directory
secondself/
├── raw/                         # Week 1: Raw captures (timestamp + unique ID)
│   └── assets/                  # Stored binary attachment assets
├── wiki/                        # Week 2: Classified + auto-linked notes
│   ├── 1_Projects/              # Active projects
│   ├── 2_Areas/                 # Key areas of responsibility
│   ├── 3_Resources/             # Reference materials & topics
│   └── 4_Archives/              # Inactive notes
├── docs/                        # Project documentation
│   ├── architecture.md          # Architectural blueprint
│   ├── Implementation-plan.md   # Phase-wise milestone plan
│   └── edge-case.md             # Edge cases & failure mitigations
├── src/                         # Core Python package
│   ├── __init__.py
│   ├── capture.py               # Ingestion logic & CLI
│   ├── classify.py              # PARA classifier (LLM)
│   ├── link.py                  # Embedding calculation & similarity auto-linker
│   ├── build_graph.py           # Nodes & edges JSON builder
│   └── ask.py                   # RAG retrieval & answer synthesizer
├── static/                      # HTML graph visualization templates
├── ProblemStatement.md          # Project problem statement
├── app.py                       # Streamlit web dashboard
└── requirements.txt             # Python dependencies
```

---

## 🛠️ Setup Instructions

### 1. Initialize Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

### 2. Set API Keys
Configure your `.env` file with `GROQ_API_KEY`, `GEMINI_API_KEY`, or `OPENAI_API_KEY`.

---

## 📜 Documentation Links
- [ProblemStatement.md](file:///d:/PersonalAssistant/ProblemStatement.md)
- [architecture.md](file:///d:/PersonalAssistant/architecture.md)
- [Implementation-plan.md](file:///d:/PersonalAssistant/Implementation-plan.md)
- [edge-case.md](file:///d:/PersonalAssistant/edge-case.md)
