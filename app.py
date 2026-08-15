"""
SecondSelf — Personal AI Second Brain | Streamlit Dashboard (Phase 5.2)

Single deployable Streamlit app with three tabs:
  Tab 1: Living Brain Visualizer — Interactive vis-network knowledge graph
  Tab 2: Ask Your Brain — RAG-powered Q&A with source citations
  Tab 3: Quick Capture — Ingest notes, URLs, and files directly from the UI
"""

import sys
import json
import os
import tempfile
from pathlib import Path
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components

# Ensure workspace root is on sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Synchronize Streamlit Cloud secrets to os.environ for backend modules
try:
    if hasattr(st, "secrets") and st.secrets:
        for key, val in st.secrets.items():
            if isinstance(val, str) and key not in os.environ:
                os.environ[key] = val
except Exception:
    pass

from src.capture import capture_item, ensure_directories
from src.classify import classify_raw_item, write_wiki_note, batch_classify
from src.link import scan_wiki_notes, run_auto_linking, load_embeddings, PARA_CATEGORIES
from src.build_graph import build_graph, export_graph_json, export_graph_html
from src.ask import ask, retrieve_relevant_notes

# Ensure runtime directories exist on fresh cloud container
ensure_directories()

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SecondSelf — AI Second Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    /* ─── UI/UX Pro Max Design Tokens ─── */
    :root {
        --bg-main: #0b0f19;
        --bg-surface: #151c2c;
        --bg-card: rgba(26, 35, 53, 0.7);
        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-hover: rgba(56, 189, 248, 0.3);
        --text-headline: #f8fafc;
        --text-body: #cbd5e1;
        --text-muted: #64748b;
        
        --brand-cyan: #38bdf8;
        --brand-indigo: #818cf8;
        --brand-purple: #c084fc;
        --gradient-brand: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        --gradient-card: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
        
        --shadow-glow: 0 0 25px -5px rgba(56, 189, 248, 0.15);
        --shadow-elevation: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }

    /* ─── Global Reset & Typography ─── */
    .stApp {
        background-color: var(--bg-main) !important;
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif !important;
        color: var(--text-body) !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: var(--text-headline) !important;
        letter-spacing: -0.02em !important;
    }

    /* ─── Header Brand Banner ─── */
    .brand-header {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.06) 0%, rgba(129, 140, 248, 0.08) 50%, rgba(192, 132, 252, 0.05) 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 20px;
        padding: 28px 36px;
        margin-bottom: 28px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        backdrop-filter: blur(16px);
        box-shadow: var(--shadow-elevation);
        position: relative;
        overflow: hidden;
    }
    .brand-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.4), transparent);
    }
    .brand-title {
        font-size: 2rem;
        font-weight: 800;
        background: var(--gradient-brand);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.03em;
        margin: 0;
    }
    .brand-subtitle {
        font-size: 0.95rem;
        color: var(--text-muted);
        margin: 6px 0 0 0;
        font-weight: 400;
    }
    .brand-badge {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.15), rgba(129, 140, 248, 0.15));
        color: var(--brand-cyan);
        padding: 8px 18px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        border: 1px solid rgba(56, 189, 248, 0.3);
        box-shadow: var(--shadow-glow);
    }

    /* ─── Stat Grid Cards ─── */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 14px;
        margin-bottom: 24px;
    }
    .stat-card {
        background: var(--gradient-card);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 18px 16px;
        text-align: center;
        backdrop-filter: blur(12px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    .stat-card:hover {
        transform: translateY(-3px);
        border-color: var(--border-hover);
        box-shadow: var(--shadow-glow);
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 800;
        background: var(--gradient-brand);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .stat-label {
        font-size: 0.72rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
        margin-top: 6px;
    }

    /* ─── Source Citation Cards ─── */
    .source-card {
        background: var(--gradient-card);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 12px;
        transition: all 0.25s ease;
        backdrop-filter: blur(8px);
    }
    .source-card:hover {
        border-color: var(--border-hover);
        transform: translateX(4px);
        box-shadow: var(--shadow-glow);
    }
    .source-title {
        font-weight: 700;
        color: var(--brand-cyan);
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .source-meta {
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-top: 6px;
    }
    .relevance-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .relevance-high { background: rgba(52, 211, 153, 0.18); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.3); }
    .relevance-med  { background: rgba(251, 191, 36, 0.18); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.3); }
    .relevance-low  { background: rgba(148, 163, 184, 0.18); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.3); }

    /* ─── Q&A Output Container ─── */
    .answer-container {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.05), rgba(56, 189, 248, 0.05));
        border: 1px solid rgba(52, 211, 153, 0.3);
        border-radius: 16px;
        padding: 24px 28px;
        margin: 20px 0;
        box-shadow: 0 10px 30px -10px rgba(16, 185, 129, 0.15);
    }

    /* ─── Capture Success Card ─── */
    .capture-success {
        background: linear-gradient(135deg, rgba(52, 211, 153, 0.12), rgba(16, 185, 129, 0.06));
        border: 1px solid rgba(52, 211, 153, 0.3);
        border-radius: 14px;
        padding: 20px 24px;
        margin-top: 14px;
        box-shadow: 0 8px 25px -5px rgba(52, 211, 153, 0.15);
    }
    .capture-success h4 {
        color: #34d399 !important;
        margin: 0 0 10px 0;
        font-weight: 700;
    }

    /* ─── Tab Styling ─── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: transparent;
        border-bottom: 1px solid var(--border-subtle);
        padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 0.92rem;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(56, 189, 248, 0.1) !important;
        color: var(--brand-cyan) !important;
        border-bottom: 2px solid var(--brand-cyan) !important;
    }

    /* ─── Category Colors ─── */
    .cat-projects  { color: #3b82f6; font-weight: 600; }
    .cat-areas     { color: #10b981; font-weight: 600; }
    .cat-resources { color: #f59e0b; font-weight: 600; }
    .cat-archives  { color: #64748b; font-weight: 600; }

    /* ─── Custom Buttons ─── */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: var(--shadow-glow) !important;
    }

    /* ─── Hide Streamlit branding ─── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    """Render the global sidebar with system info and quick actions."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 16px 0;">
            <div style="font-size: 2.5rem; margin-bottom: 4px;">🧠</div>
            <div style="font-size: 1.1rem; font-weight: 700;
                        background: linear-gradient(135deg, #38bdf8, #818cf8);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                SecondSelf
            </div>
            <div style="font-size: 0.75rem; color: #64748b;">Personal AI Second Brain</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # System Status
        st.markdown("##### 📊 System Status")
        notes = scan_wiki_notes()
        embeddings = load_embeddings()

        col1, col2 = st.columns(2)
        col1.metric("Wiki Notes", len(notes))

        emb_count = 0
        if embeddings and embeddings.get("notes"):
            emb_count = len(embeddings["notes"])
        col2.metric("Indexed", emb_count)

        # Per-category breakdown
        cat_counts = {}
        for note in notes:
            cat = note.get("category", "Unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        cat_icons = {
            "1_Projects": "🔵",
            "2_Areas": "🟢",
            "3_Resources": "🟡",
            "4_Archives": "⚪",
        }
        for cat in PARA_CATEGORIES:
            count = cat_counts.get(cat, 0)
            icon = cat_icons.get(cat, "📄")
            st.caption(f"{icon} {cat.replace('_', ' ')}: **{count}**")

        st.divider()

        # Pipeline Actions
        st.markdown("##### ⚡ Pipeline Actions")

        if st.button("🔄 Rebuild Embeddings", use_container_width=True, key="sidebar_rebuild_emb"):
            with st.spinner("Computing embeddings..."):
                try:
                    run_auto_linking()
                    st.success("[OK] Embeddings rebuilt & notes re-linked!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        if st.button("🗺️ Rebuild Graph", use_container_width=True, key="sidebar_rebuild_graph"):
            with st.spinner("Building knowledge graph..."):
                try:
                    graph = build_graph()
                    export_graph_json(graph)
                    export_graph_html(graph)
                    st.success(f"[OK] Graph rebuilt: {graph['metadata']['total_nodes']} nodes, {graph['metadata']['total_edges']} edges")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        if st.button("🏷️ Classify Raw Items", use_container_width=True, key="sidebar_classify"):
            with st.spinner("Classifying unprocessed captures..."):
                try:
                    created = batch_classify()
                    st.success(f"[OK] Classified {len(created)} items")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        st.divider()

        # LLM Provider Status
        st.markdown("##### 🔑 LLM Providers")
        for name, env_var in [("Groq", "GROQ_API_KEY"), ("Gemini", "GEMINI_API_KEY"), ("OpenAI", "OPENAI_API_KEY")]:
            key = os.getenv(env_var, "")
            if key and not key.startswith("your_"):
                st.caption(f"✅ {name}: configured")
            else:
                st.caption(f"❌ {name}: not set")


# ---------------------------------------------------------------------------
# Tab 1: Living Brain Visualizer
# ---------------------------------------------------------------------------

def render_brain_tab():
    """Render the interactive knowledge graph visualization tab."""

    # Load graph data
    graph_json_path = BASE_DIR / "graph.json"
    graph_html_path = BASE_DIR / "static" / "graph.html"

    if not graph_json_path.exists():
        st.warning("⚠️ No graph.json found. Click **Rebuild Graph** in the sidebar first.")
        if st.button("🗺️ Build Graph Now", key="build_graph_brain"):
            with st.spinner("Building knowledge graph..."):
                graph = build_graph()
                export_graph_json(graph)
                export_graph_html(graph)
                st.success("[OK] Graph built!")
                st.rerun()
        return

    with open(graph_json_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    metadata = graph_data.get("metadata", {})

    # Stat cards
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-value">{metadata.get('total_nodes', 0)}</div>
            <div class="stat-label">Total Nodes</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{metadata.get('node_counts', {}).get('notes', 0)}</div>
            <div class="stat-label">Notes</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{metadata.get('node_counts', {}).get('tags', 0)}</div>
            <div class="stat-label">Tags</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{metadata.get('total_edges', 0)}</div>
            <div class="stat-label">Connections</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{metadata.get('edge_counts', {}).get('semantic', 0)}</div>
            <div class="stat-label">Semantic Links</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{metadata.get('edge_counts', {}).get('explicit', 0)}</div>
            <div class="stat-label">Wikilinks</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Render interactive graph via embedded HTML
    if graph_html_path.exists():
        graph_html = graph_html_path.read_text(encoding="utf-8")
        components.html(graph_html, height=650, scrolling=False)
    else:
        # Fallback: build inline vis-network from graph.json
        _render_inline_graph(graph_data)

    # Graph legend
    with st.expander("📖 Graph Legend & Tips", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **PARA Categories**
            - 🔵 **Projects** — Active goals & deadlines
            - 🟢 **Areas** — Ongoing responsibilities
            - 🟡 **Resources** — Reference material
            - ⚪ **Archives** — Inactive items
            """)
        with col2:
            st.markdown("""
            **Node Types**
            - 🟣 **Tag nodes** — Shared topic labels
            - 💎 **Category nodes** — PARA groupings

            **Interactions**: Drag nodes, scroll to zoom, hover for previews
            """)


def _render_inline_graph(graph_data):
    """Fallback inline vis-network renderer if static HTML is missing."""
    json_str = json.dumps(graph_data, ensure_ascii=False)

    html = f"""
    <!DOCTYPE html>
    <html><head>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{ margin:0; background: #0f172a; }}
        #graph {{ width:100%; height:620px; }}
    </style>
    </head><body>
    <div id="graph"></div>
    <script>
    const raw = {json_str};
    const colorMap = {{
        '1_Projects': '#3b82f6', '2_Areas': '#10b981',
        '3_Resources': '#f59e0b', '4_Archives': '#64748b',
        'tags': '#a855f7'
    }};
    const shapeMap = {{ 'note': 'dot', 'category': 'diamond', 'tag': 'triangle' }};
    const nodes = new vis.DataSet(raw.nodes.map(n => ({{
        id: n.id, label: n.label, group: n.group,
        shape: shapeMap[n.type] || 'dot',
        color: {{ background: colorMap[n.group] || '#94a3b8', border: colorMap[n.group] || '#94a3b8' }},
        size: n.type === 'note' ? 18 : n.type === 'category' ? 24 : 12,
        title: '<b>' + n.label + '</b>' + (n.summary ? '<br>' + n.summary : ''),
        font: {{ color: '#e2e8f0', size: n.type === 'note' ? 12 : 10 }}
    }})));
    const edgeColorMap = {{ 'category': '#334155', 'tag': '#1e293b', 'explicit': '#60a5fa', 'semantic': '#a78bfa' }};
    const edges = new vis.DataSet(raw.edges.map(e => ({{
        from: e.from, to: e.to,
        color: {{ color: edgeColorMap[e.type] || '#334155', opacity: e.type === 'tag' ? 0.3 : 0.6 }},
        width: e.type === 'semantic' ? 2 : 1,
        smooth: {{ type: 'continuous' }}
    }})));
    new vis.Network(document.getElementById('graph'), {{ nodes, edges }}, {{
        physics: {{ barnesHut: {{ gravitationalConstant: -3000, springLength: 120 }}, stabilization: {{ iterations: 150 }} }},
        interaction: {{ hover: true, tooltipDelay: 100, zoomView: true, dragView: true }},
    }});
    </script></body></html>
    """
    components.html(html, height=640, scrolling=False)


# ---------------------------------------------------------------------------
# Tab 2: Ask Your Brain
# ---------------------------------------------------------------------------

def render_ask_tab():
    """Render the RAG Q&A search tab."""

    st.markdown("""
    <div style="text-align: center; margin-bottom: 24px;">
        <div style="font-size: 2.4rem;">🔮</div>
        <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 4px;">
            Ask questions in natural language — answers are grounded in your personal notes.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Search form
    col1, col2, col3 = st.columns([5, 1, 1])
    with col1:
        question = st.text_input(
            "Ask your brain",
            placeholder="e.g. What AI projects am I working on?",
            label_visibility="collapsed",
            key="ask_question",
        )
    with col2:
        top_k = st.selectbox("Results", [3, 5, 7, 10], index=1, key="ask_topk")
    with col3:
        provider = st.selectbox("LLM", ["auto", "groq", "gemini", "openai"], index=0, key="ask_provider")

    search_clicked = st.button("🔍  Ask Your Brain", type="primary", use_container_width=True, key="ask_submit")

    if search_clicked and question.strip():
        with st.spinner("🔮 Searching your knowledge base and synthesizing answer..."):
            try:
                provider_arg = None if provider == "auto" else provider
                result = ask(
                    question=question.strip(),
                    top_k=top_k,
                    provider=provider_arg,
                )

                # Display answer
                st.markdown("---")
                st.markdown("### 🔮 Answer")
                st.markdown(result["answer"])

                # Display sources
                if result["sources"]:
                    st.markdown("---")
                    st.markdown("### 📚 Sources")

                    for i, src in enumerate(result["sources"], 1):
                        score = src["similarity_score"]
                        if score >= 0.6:
                            badge_class = "relevance-high"
                        elif score >= 0.35:
                            badge_class = "relevance-med"
                        else:
                            badge_class = "relevance-low"

                        cat_class = f"cat-{src['category'].split('_')[1].lower()}" if '_' in src['category'] else ""

                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-title">
                                {i}. {src['title']}
                                <span class="relevance-badge {badge_class}">{score:.0%} match</span>
                            </div>
                            <div class="source-meta">
                                <span class="{cat_class}">{src['category'].replace('_', ' ')}</span>
                                &nbsp;·&nbsp; {Path(src['path']).name}
                                &nbsp;·&nbsp; Tags: {', '.join(src.get('tags', [])[:3])}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                # Footer
                st.caption(f"Provider: **{result['provider']}** · Notes retrieved: **{result['retrieval_count']}**")

            except Exception as e:
                st.error(f"Error during Q&A: {e}")

    elif search_clicked:
        st.warning("Please enter a question.")

    # Recent knowledge overview
    st.markdown("---")
    with st.expander("📋 Knowledge Base Overview", expanded=False):
        notes = scan_wiki_notes()
        if notes:
            for cat in PARA_CATEGORIES:
                cat_notes = [n for n in notes if n["category"] == cat]
                if cat_notes:
                    st.markdown(f"**{cat.replace('_', ' ')}** ({len(cat_notes)} notes)")
                    for note in cat_notes[:5]:
                        summary = note.get("summary", "")[:80]
                        st.caption(f"  • {note['title']} — {summary}")
                    if len(cat_notes) > 5:
                        st.caption(f"  *...and {len(cat_notes) - 5} more*")
        else:
            st.info("No notes yet. Use the **Quick Capture** tab to add your first notes!")


# ---------------------------------------------------------------------------
# Tab 3: Quick Capture
# ---------------------------------------------------------------------------

def render_capture_tab():
    """Render the quick capture interface for notes, URLs, and files."""

    st.markdown("""
    <div style="text-align: center; margin-bottom: 24px;">
        <div style="font-size: 2.4rem;">📥</div>
        <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 4px;">
            Capture notes, web links, or files — they'll be classified and indexed automatically.
        </p>
    </div>
    """, unsafe_allow_html=True)

    ensure_directories()

    capture_type = st.radio(
        "What would you like to capture?",
        ["📝 Text Note", "🔗 Web Link", "📄 File Upload"],
        horizontal=True,
        key="capture_type",
    )

    st.markdown("---")

    # ── Text Note ──
    if capture_type == "📝 Text Note":
        title = st.text_input("Note Title (optional)", key="note_title",
                              placeholder="e.g. Meeting notes — Sprint Planning")
        content = st.text_area("Note Content", height=200, key="note_content",
                               placeholder="Write your thoughts, ideas, or notes here...")

        auto_classify = st.checkbox("🤖 Auto-classify after capture", value=True, key="note_auto_classify")

        if st.button("💾 Capture Note", type="primary", use_container_width=True, key="note_submit"):
            if not content.strip():
                st.warning("Please enter some note content.")
            else:
                with st.spinner("Capturing note..."):
                    try:
                        record = capture_item("note", content.strip(), title=title.strip() or None)
                        _show_capture_success(record, auto_classify)
                    except Exception as e:
                        st.error(f"Capture failed: {e}")

    # ── Web Link ──
    elif capture_type == "🔗 Web Link":
        url = st.text_input("URL", key="link_url",
                            placeholder="https://example.com/interesting-article")
        title = st.text_input("Title Override (optional)", key="link_title",
                              placeholder="Leave blank to auto-extract from page")

        auto_classify = st.checkbox("🤖 Auto-classify after capture", value=True, key="link_auto_classify")

        if st.button("🌐 Capture Link", type="primary", use_container_width=True, key="link_submit"):
            if not url.strip():
                st.warning("Please enter a URL.")
            else:
                with st.spinner("Scraping and capturing link..."):
                    try:
                        record = capture_item("link", url.strip(), title=title.strip() or None)
                        _show_capture_success(record, auto_classify)
                    except Exception as e:
                        st.error(f"Capture failed: {e}")

    # ── File Upload ──
    elif capture_type == "📄 File Upload":
        uploaded_file = st.file_uploader(
            "Upload a file (PDF, TXT, MD, or any text document)",
            type=["pdf", "txt", "md", "py", "json", "csv", "html"],
            key="file_upload",
        )
        title = st.text_input("Title Override (optional)", key="file_title",
                              placeholder="Leave blank to use filename")

        auto_classify = st.checkbox("🤖 Auto-classify after capture", value=True, key="file_auto_classify")

        if st.button("📤 Capture File", type="primary", use_container_width=True, key="file_submit"):
            if uploaded_file is None:
                st.warning("Please upload a file first.")
            else:
                with st.spinner("Processing and capturing file..."):
                    try:
                        # Save uploaded file to a temp location
                        suffix = Path(uploaded_file.name).suffix
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(BASE_DIR)) as tmp:
                            tmp.write(uploaded_file.getbuffer())
                            tmp_path = tmp.name

                        record = capture_item("file", tmp_path, title=title.strip() or None)

                        # Clean up temp file
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass

                        _show_capture_success(record, auto_classify)

                    except Exception as e:
                        st.error(f"Capture failed: {e}")


def _show_capture_success(record: dict, auto_classify: bool):
    """Display capture success and optionally auto-classify."""

    st.markdown(f"""
    <div class="capture-success">
        <h4>✅ Captured Successfully!</h4>
        <p style="margin: 0; color: #94a3b8; font-size: 0.88rem;">
            <strong>ID:</strong> {record.get('id', 'N/A')}<br>
            <strong>Title:</strong> {record.get('title', 'Untitled')}<br>
            <strong>Type:</strong> {record.get('type', 'note').upper()}<br>
            <strong>Time:</strong> {record.get('timestamp', '')}
        </p>
    </div>
    """, unsafe_allow_html=True)

    if auto_classify:
        st.markdown("---")
        with st.spinner("🤖 Auto-classifying with AI..."):
            try:
                classification = classify_raw_item(record)
                wiki_path = write_wiki_note(record, classification)

                cat_icon = {
                    "1_Projects": "🔵", "2_Areas": "🟢",
                    "3_Resources": "🟡", "4_Archives": "⚪",
                }.get(classification["category"], "📄")

                st.success(
                    f"Classified → {cat_icon} **{classification['category'].replace('_', ' ')}**\n\n"
                    f"**Title:** {classification['title']}\n\n"
                    f"**Tags:** {', '.join(classification['tags'])}\n\n"
                    f"**Summary:** {classification['summary']}\n\n"
                    f"**Provider:** {classification.get('provider', 'unknown')}"
                )

            except Exception as e:
                st.warning(f"Auto-classification failed: {e}\n\nThe raw capture was still saved. "
                           "You can classify it later using the sidebar.")


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

def main():
    """Main Streamlit application entrypoint."""

    render_sidebar()

    # Brand Header
    st.markdown("""
    <div class="brand-header">
        <div>
            <h1 class="brand-title">🧠 SecondSelf</h1>
            <p class="brand-subtitle">Your Personal AI Second Brain — Capture, Organize, Visualize, Ask</p>
        </div>
        <span class="brand-badge">🏅 The Oracle</span>
    </div>
    """, unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "🗺️  Living Brain",
        "🔮  Ask Your Brain",
        "📥  Quick Capture",
    ])

    with tab1:
        render_brain_tab()

    with tab2:
        render_ask_tab()

    with tab3:
        render_capture_tab()


# Streamlit Cloud runs `streamlit run app.py` which does NOT trigger __main__.
# Call main() unconditionally so the app renders in all execution contexts.
main()
