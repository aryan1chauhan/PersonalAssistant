"""
SecondSelf Knowledge Graph Data Model Exporter — "The Cartographer" (Phase 4.1)

Parses all wiki notes, extracts nodes (notes, categories, tags) and edges
(explicit wikilinks, semantic similarity, category membership, tag membership),
and exports the full graph as a structured graph.json file.
"""

import os
import sys
import json
import re
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Set

import click
from rich.console import Console
from rich.table import Table
from dotenv import load_dotenv

# Ensure workspace root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.link import (
    scan_wiki_notes,
    parse_wiki_note,
    load_embeddings,
    cosine_similarity,
    PARA_CATEGORIES,
    RELATED_SECTION_HEADER,
    SIMILARITY_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()

# Force UTF-8 stdout on Windows to prevent cp1252 encoding errors with Rich.
def _fix_windows_encoding():
    if sys.platform == "win32":
        try:
            if getattr(sys.stdout, "encoding", "").lower() != "utf-8":
                import io
                sys.stdout = io.TextIOWrapper(getattr(sys.stdout, "buffer", sys.stdout), encoding="utf-8", errors="replace")
                sys.stderr = io.TextIOWrapper(getattr(sys.stderr, "buffer", sys.stderr), encoding="utf-8", errors="replace")
        except Exception:
            pass

_fix_windows_encoding()
console = Console()

BASE_DIR = Path(__file__).resolve().parent.parent
WIKI_DIR = BASE_DIR / "wiki"
DATA_DIR = BASE_DIR / "data"
GRAPH_JSON_PATH = BASE_DIR / "graph.json"


# ---------------------------------------------------------------------------
# Node Builders
# ---------------------------------------------------------------------------

def _note_slug(note: Dict[str, Any]) -> str:
    """Generate a stable slug ID for a note node from its file path."""
    filepath = note["path"]
    # Use category/filename_stem as unique ID
    return f"{filepath.parent.name}/{filepath.stem}"


def build_note_nodes(notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build note-type graph nodes from parsed wiki notes.

    Each note becomes a node with:
      id, label, group (PARA category), type="note", summary, tags, word_count, file_path.
    """
    nodes = []
    for note in notes:
        node = {
            "id": _note_slug(note),
            "label": note["title"],
            "group": note["category"],
            "type": "note",
            "summary": note.get("summary", ""),
            "tags": note.get("tags", []),
            "word_count": note.get("word_count", 0),
            "file_path": str(note["path"]),
        }
        nodes.append(node)
    return nodes


def build_category_nodes(notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build category-type graph nodes — one per PARA category that has at least one note.
    """
    populated_categories: Set[str] = set()
    for note in notes:
        cat = note.get("category", "")
        if cat:
            populated_categories.add(cat)

    nodes = []
    for cat in sorted(populated_categories):
        nodes.append({
            "id": f"category/{cat}",
            "label": cat.replace("_", " "),
            "group": cat,
            "type": "category",
        })
    return nodes


def build_tag_nodes(notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build tag-type graph nodes — one per unique tag across all notes.
    """
    all_tags: Set[str] = set()
    for note in notes:
        for tag in note.get("tags", []):
            tag_clean = tag.strip().lower()
            if tag_clean:
                all_tags.add(tag_clean)

    nodes = []
    for tag in sorted(all_tags):
        nodes.append({
            "id": f"tag/{tag}",
            "label": tag,
            "group": "tags",
            "type": "tag",
        })
    return nodes


# ---------------------------------------------------------------------------
# Edge Builders
# ---------------------------------------------------------------------------

def build_category_edges(notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build edges connecting each note to its PARA category node.
    """
    edges = []
    for note in notes:
        cat = note.get("category", "")
        if cat:
            edges.append({
                "from": _note_slug(note),
                "to": f"category/{cat}",
                "type": "category",
                "weight": 1.0,
            })
    return edges


def build_tag_edges(notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build edges connecting each note to its tag nodes.
    """
    edges = []
    for note in notes:
        for tag in note.get("tags", []):
            tag_clean = tag.strip().lower()
            if tag_clean:
                edges.append({
                    "from": _note_slug(note),
                    "to": f"tag/{tag_clean}",
                    "type": "tag",
                    "weight": 1.0,
                })
    return edges


def _extract_wikilink_titles(filepath: Path) -> List[str]:
    """Extract [[wikilink]] titles from the ## Related Knowledge section of a note."""
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception:
        return []

    # Split on the Related Knowledge header and only look at that section
    parts = re.split(r"^## Related Knowledge\s*$", content, flags=re.MULTILINE)
    if len(parts) < 2:
        return []

    related_section = parts[1]
    return re.findall(r"\[\[([^\]]+)\]\]", related_section)


def build_wikilink_edges(
    notes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build edges from explicit [[wikilinks]] in ## Related Knowledge sections.

    Only creates edges where both the source and target notes exist.
    Dangling links (pointing to non-existent notes) are skipped.
    """
    # Build title → slug lookup for resolving wikilink targets
    title_to_slug: Dict[str, str] = {}
    for note in notes:
        title_to_slug[note["title"]] = _note_slug(note)

    edges = []
    seen_pairs: Set[Tuple[str, str]] = set()

    for note in notes:
        source_slug = _note_slug(note)
        linked_titles = _extract_wikilink_titles(note["path"])

        for target_title in linked_titles:
            target_slug = title_to_slug.get(target_title)
            if target_slug is None:
                continue  # Dangling link — target note doesn't exist
            if target_slug == source_slug:
                continue  # Self-link — skip

            # Deduplicate: normalize pair order so A→B and B→A become one edge
            pair = tuple(sorted([source_slug, target_slug]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            edges.append({
                "from": source_slug,
                "to": target_slug,
                "type": "explicit",
                "weight": 1.0,
            })

    return edges


def build_semantic_edges(
    notes: List[Dict[str, Any]],
    data_dir: Optional[Path] = None,
    threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Build edges from semantic similarity data stored in data/embeddings.pkl.

    Loads pre-computed embeddings and finds all note pairs above the similarity
    threshold. Returns empty list if embeddings file doesn't exist.
    """
    target_dir = data_dir or DATA_DIR
    sim_threshold = threshold if threshold is not None else SIMILARITY_THRESHOLD

    emb_data = load_embeddings(target_dir)
    if emb_data is None:
        return []

    emb_notes = emb_data.get("notes", [])
    if len(emb_notes) < 2:
        return []

    # Build path → slug lookup for current notes
    path_to_slug: Dict[str, str] = {}
    for note in notes:
        path_to_slug[str(note["path"])] = _note_slug(note)

    # Build path → embedding lookup from stored data
    path_to_emb: Dict[str, Any] = {}
    for rec in emb_notes:
        path_to_emb[rec["path"]] = rec["embedding"]

    # Find all note paths that exist in both current wiki and stored embeddings
    common_paths = [p for p in path_to_slug if p in path_to_emb]
    if len(common_paths) < 2:
        return []

    edges = []
    seen_pairs: Set[Tuple[str, str]] = set()

    for i, path_a in enumerate(common_paths):
        for j in range(i + 1, len(common_paths)):
            path_b = common_paths[j]
            sim = cosine_similarity(path_to_emb[path_a], path_to_emb[path_b])

            if sim >= sim_threshold:
                slug_a = path_to_slug[path_a]
                slug_b = path_to_slug[path_b]
                pair = tuple(sorted([slug_a, slug_b]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    edges.append({
                        "from": slug_a,
                        "to": slug_b,
                        "type": "semantic",
                        "weight": round(float(sim), 4),
                    })

    return edges


# ---------------------------------------------------------------------------
# Full Graph Builder
# ---------------------------------------------------------------------------

def build_graph(
    wiki_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build the complete knowledge graph data model.

    Returns a dict with 'metadata', 'nodes', and 'edges' keys.
    """
    target_wiki = wiki_dir or WIKI_DIR

    # Parse all wiki notes
    notes = scan_wiki_notes(target_wiki)

    # Build nodes
    note_nodes = build_note_nodes(notes)
    category_nodes = build_category_nodes(notes)
    tag_nodes = build_tag_nodes(notes)
    all_nodes = note_nodes + category_nodes + tag_nodes

    # Build edges
    category_edges = build_category_edges(notes)
    tag_edges = build_tag_edges(notes)
    wikilink_edges = build_wikilink_edges(notes)
    semantic_edges = build_semantic_edges(notes, data_dir=data_dir, threshold=threshold)
    all_edges = category_edges + tag_edges + wikilink_edges + semantic_edges

    graph = {
        "metadata": {
            "generated_at": datetime.now().astimezone().isoformat(),
            "total_nodes": len(all_nodes),
            "total_edges": len(all_edges),
            "node_counts": {
                "notes": len(note_nodes),
                "categories": len(category_nodes),
                "tags": len(tag_nodes),
            },
            "edge_counts": {
                "category": len(category_edges),
                "tag": len(tag_edges),
                "explicit": len(wikilink_edges),
                "semantic": len(semantic_edges),
            },
        },
        "nodes": all_nodes,
        "edges": all_edges,
    }

    return graph


STATIC_DIR = BASE_DIR / "static"
GRAPH_HTML_PATH = STATIC_DIR / "graph.html"


def export_graph_json(
    graph: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Write the graph data model to a JSON file.
    Returns the path of the written file.
    """
    target_path = output_path or GRAPH_JSON_PATH

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False, default=str)

    return target_path


def export_graph_html(
    graph: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Export an interactive standalone HTML graph visualization using vis-network.
    """
    target_path = output_path or GRAPH_HTML_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    json_data_str = json.dumps(graph, indent=2, ensure_ascii=False)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SecondSelf — Knowledge Graph Visualizer</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
            height: 100vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }}
        header {{
            background: rgba(30, 41, 59, 0.85);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding: 12px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            z-index: 10;
        }}
        .brand {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .brand h1 {{
            font-size: 1.1rem;
            font-weight: 700;
            background: linear-gradient(135deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.02em;
        }}
        .badge {{
            background: rgba(255, 255, 255, 0.08);
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 500;
            color: #94a3b8;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }}
        .stats {{
            display: flex;
            gap: 16px;
        }}
        .stat-item {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.8rem;
            color: #cbd5e1;
        }}
        .stat-val {{
            font-weight: 600;
            color: #38bdf8;
        }}
        .controls {{
            display: flex;
            gap: 12px;
            align-items: center;
        }}
        input[type="text"] {{
            background: #1e293b;
            border: 1px solid #334155;
            color: #f8fafc;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.85rem;
            outline: none;
            width: 220px;
            transition: all 0.2s;
        }}
        input[type="text"]:focus {{
            border-color: #38bdf8;
            box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2);
        }}
        button {{
            background: #334155;
            border: none;
            color: #f8fafc;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
        }}
        button:hover {{
            background: #475569;
        }}
        #mynetwork {{
            flex: 1;
            width: 100%;
            height: 100%;
            background: radial-gradient(circle at 50% 50%, #1e293b 0%, #0f172a 100%);
        }}
        .legend {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(30, 41, 59, 0.9);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 14px;
            border-radius: 10px;
            font-size: 0.78rem;
            display: flex;
            flex-direction: column;
            gap: 8px;
            z-index: 5;
            max-width: 240px;
        }}
        .legend-title {{
            font-weight: 600;
            color: #94a3b8;
            margin-bottom: 4px;
            text-transform: uppercase;
            font-size: 0.68rem;
            letter-spacing: 0.05em;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }}
        div.vis-tooltip {{
            background-color: rgba(15, 23, 42, 0.95) !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5) !important;
            color: #f8fafc !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.85rem !important;
            border-radius: 8px !important;
            padding: 12px 14px !important;
            max-width: 320px !important;
            line-height: 1.4 !important;
        }}
    </style>
</head>
<body>

    <header>
        <div class="brand">
            <h1>SecondSelf</h1>
            <span class="badge">Living Brain Graph</span>
        </div>
        <div class="stats">
            <div class="stat-item">Nodes: <span class="stat-val" id="stat-nodes">0</span></div>
            <div class="stat-item">Notes: <span class="stat-val" id="stat-notes">0</span></div>
            <div class="stat-item">Categories: <span class="stat-val" id="stat-cats">0</span></div>
            <div class="stat-item">Tags: <span class="stat-val" id="stat-tags">0</span></div>
            <div class="stat-item">Edges: <span class="stat-val" id="stat-edges">0</span></div>
        </div>
        <div class="controls">
            <input type="text" id="search-input" placeholder="Search nodes..." onkeyup="searchNodes()">
            <button onclick="resetView()">Reset View</button>
            <button onclick="togglePhysics()" id="physics-btn">Pause Physics</button>
        </div>
    </header>

    <div id="mynetwork"></div>

    <div class="legend">
        <div class="legend-title">PARA Categories</div>
        <div class="legend-item"><div class="dot" style="background:#3b82f6;"></div> 1_Projects (Active)</div>
        <div class="legend-item"><div class="dot" style="background:#10b981;"></div> 2_Areas (Responsibility)</div>
        <div class="legend-item"><div class="dot" style="background:#f59e0b;"></div> 3_Resources (Reference)</div>
        <div class="legend-item"><div class="dot" style="background:#64748b;"></div> 4_Archives (Inactive)</div>
        <div class="legend-title" style="margin-top:6px;">Node Types</div>
        <div class="legend-item"><div class="dot" style="background:#a855f7;"></div> Tag Node</div>
        <div class="legend-item"><div class="dot" style="background:#ec4899; transform:rotate(45deg); border-radius:2px;"></div> Category Node</div>
    </div>

    <script type="text/javascript">
        const rawData = {json_data_str};

        // Populate statistics
        document.getElementById('stat-nodes').innerText = rawData.metadata.total_nodes;
        document.getElementById('stat-notes').innerText = rawData.metadata.node_counts.notes;
        document.getElementById('stat-cats').innerText = rawData.metadata.node_counts.categories;
        document.getElementById('stat-tags').innerText = rawData.metadata.node_counts.tags;
        document.getElementById('stat-edges').innerText = rawData.metadata.total_edges;

        const categoryColors = {{
            '1_Projects': '#3b82f6',
            '2_Areas': '#10b981',
            '3_Resources': '#f59e0b',
            '4_Archives': '#64748b'
        }};

        // Process Vis-Network Nodes
        const nodesArray = rawData.nodes.map(n => {{
            let nodeObj = {{
                id: n.id,
                label: n.label,
                title: `<div style="font-weight:600; font-size:0.95rem; margin-bottom:4px; color:#38bdf8;">${{n.label}}</div>
                        <div style="font-size:0.75rem; color:#94a3b8; margin-bottom:6px;">Group: ${{n.group}} | Type: ${{n.type}}</div>
                        ${{n.summary ? `<div style="font-size:0.8rem; margin-bottom:6px;">${{n.summary}}</div>` : ''}}
                        ${{n.tags && n.tags.length ? `<div style="font-size:0.72rem; color:#a855f7;">Tags: ${{n.tags.join(', ')}}</div>` : ''}}`
            }};

            if (n.type === 'note') {{
                const color = categoryColors[n.group] || '#3b82f6';
                nodeObj.color = {{
                    background: color,
                    border: '#ffffff',
                    highlight: {{ background: '#38bdf8', border: '#ffffff' }}
                }};
                nodeObj.shape = 'dot';
                nodeObj.size = 20 + Math.min(n.word_count / 10, 15);
                nodeObj.font = {{ color: '#f8fafc', size: 12, face: 'Inter' }};
            }} else if (n.type === 'category') {{
                nodeObj.color = {{
                    background: '#ec4899',
                    border: '#f472b6',
                    highlight: {{ background: '#f472b6', border: '#ffffff' }}
                }};
                nodeObj.shape = 'diamond';
                nodeObj.size = 30;
                nodeObj.font = {{ color: '#f472b6', size: 14, weight: 'bold', face: 'Inter' }};
            }} else if (n.type === 'tag') {{
                nodeObj.color = {{
                    background: '#a855f7',
                    border: '#c084fc',
                    highlight: {{ background: '#c084fc', border: '#ffffff' }}
                }};
                nodeObj.shape = 'dot';
                nodeObj.size = 12;
                nodeObj.font = {{ color: '#c084fc', size: 11, face: 'Inter' }};
            }}

            return nodeObj;
        }});

        // Process Vis-Network Edges
        const edgesArray = rawData.edges.map(e => {{
            let edgeObj = {{
                from: e.from,
                to: e.to,
                width: 1
            }};

            if (e.type === 'explicit') {{
                edgeObj.color = {{ color: '#38bdf8', highlight: '#7dd3fc' }};
                edgeObj.width = 2.5;
                edgeObj.title = 'Wikilink';
            }} else if (e.type === 'semantic') {{
                edgeObj.color = {{ color: '#22c55e', highlight: '#4ade80' }};
                edgeObj.width = 2;
                edgeObj.dashes = [4, 4];
                edgeObj.title = `Similarity: ${{e.weight}}`;
            }} else if (e.type === 'category') {{
                edgeObj.color = {{ color: 'rgba(236, 72, 153, 0.3)', highlight: '#ec4899' }};
                edgeObj.width = 1;
            }} else if (e.type === 'tag') {{
                edgeObj.color = {{ color: 'rgba(168, 85, 247, 0.25)', highlight: '#a855f7' }};
                edgeObj.width = 1;
            }}

            return edgeObj;
        }});

        const container = document.getElementById('mynetwork');
        const data = {{
            nodes: new vis.DataSet(nodesArray),
            edges: new vis.DataSet(edgesArray)
        }};

        const options = {{
            nodes: {{
                borderWidth: 1.5,
                shadow: true
            }},
            edges: {{
                smooth: {{
                    type: 'continuous',
                    forceDirection: 'none'
                }}
            }},
            physics: {{
                solver: 'barnesHut',
                barnesHut: {{
                    gravitationalConstant: -3000,
                    centralGravity: 0.3,
                    springLength: 95,
                    springConstant: 0.04,
                    damping: 0.09
                }},
                stabilization: {{
                    enabled: true,
                    iterations: 150
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100,
                zoomView: true,
                dragView: true
            }}
        }};

        const network = new vis.Network(container, data, options);

        let physicsEnabled = true;
        function togglePhysics() {{
            physicsEnabled = !physicsEnabled;
            network.setOptions({{ physics: {{ enabled: physicsEnabled }} }});
            document.getElementById('physics-btn').innerText = physicsEnabled ? 'Pause Physics' : 'Resume Physics';
        }}

        function resetView() {{
            network.fit({{ animation: {{ duration: 500, easingFunction: 'easeInOutQuad' }} }});
        }}

        function searchNodes() {{
            const query = document.getElementById('search-input').value.toLowerCase().trim();
            if (!query) return;

            const matches = nodesArray.filter(n => n.label.toLowerCase().includes(query));
            if (matches.length > 0) {{
                network.selectNodes([matches[0].id]);
                network.focus(matches[0].id, {{ scale: 1.2, animation: {{ duration: 500 }} }});
            }}
        }}
    </script>
</body>
</html>
"""

    with open(target_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return target_path


# ---------------------------------------------------------------------------
# Click CLI
# ---------------------------------------------------------------------------

@click.group()
def main():
    """SecondSelf Graph Builder — Export wiki notes as a knowledge graph."""
    pass


@main.command("run")
@click.option(
    "--output", "-o", type=click.Path(), default=None,
    help=f"Output path for graph.json (default: {GRAPH_JSON_PATH}).",
)
@click.option(
    "--threshold", "-t", type=float, default=None,
    help=f"Cosine similarity threshold for semantic edges (default: {SIMILARITY_THRESHOLD}).",
)
def run_cmd(output: Optional[str], threshold: Optional[float]):
    """Build the knowledge graph and export graph.json and static/graph.html."""
    console.print("\n[bold magenta]>> The Cartographer — Knowledge Graph Builder[/bold magenta]\n")

    console.print("[bold blue]Step 1/3:[/bold blue] Building graph data model...")
    graph = build_graph(threshold=threshold)

    meta = graph["metadata"]
    console.print(f"  Nodes: [cyan]{meta['total_nodes']}[/cyan] "
                  f"({meta['node_counts']['notes']} notes, "
                  f"{meta['node_counts']['categories']} categories, "
                  f"{meta['node_counts']['tags']} tags)")
    console.print(f"  Edges: [cyan]{meta['total_edges']}[/cyan] "
                  f"({meta['edge_counts']['explicit']} wikilinks, "
                  f"{meta['edge_counts']['semantic']} semantic, "
                  f"{meta['edge_counts']['category']} category, "
                  f"{meta['edge_counts']['tag']} tag)")

    console.print("\n[bold blue]Step 2/3:[/bold blue] Exporting graph.json...")
    out_path = Path(output) if output else None
    result_path = export_graph_json(graph, output_path=out_path)
    console.print(f"  Saved to [cyan]{result_path}[/cyan]")

    console.print("\n[bold blue]Step 3/3:[/bold blue] Exporting interactive HTML visualizer...")
    html_path = export_graph_html(graph)
    console.print(f"  Saved to [cyan]{html_path}[/cyan]")

    console.print(f"\n[bold green][OK] Knowledge graph and HTML visualizer exported successfully![/bold green]\n")


@main.command("status")
def status_cmd():
    """Show graph statistics from existing graph.json."""
    if not GRAPH_JSON_PATH.exists():
        console.print("[bold yellow]No graph.json found. Run 'python build_graph.py run' first.[/bold yellow]")
        return

    try:
        with open(GRAPH_JSON_PATH, "r", encoding="utf-8") as f:
            graph = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        console.print(f"[bold red]Error reading graph.json: {e}[/bold red]")
        return

    meta = graph.get("metadata", {})
    console.print(f"\n[bold]Knowledge Graph Status — graph.json[/bold]")
    console.print(f"  Generated at:  {meta.get('generated_at', 'unknown')}")
    console.print(f"  Total nodes:   {meta.get('total_nodes', 0)}")
    console.print(f"  Total edges:   {meta.get('total_edges', 0)}")

    node_counts = meta.get("node_counts", {})
    console.print(f"    Notes:       {node_counts.get('notes', 0)}")
    console.print(f"    Categories:  {node_counts.get('categories', 0)}")
    console.print(f"    Tags:        {node_counts.get('tags', 0)}")

    edge_counts = meta.get("edge_counts", {})
    console.print(f"    Wikilinks:   {edge_counts.get('explicit', 0)}")
    console.print(f"    Semantic:    {edge_counts.get('semantic', 0)}")
    console.print(f"    Category:    {edge_counts.get('category', 0)}")
    console.print(f"    Tag:         {edge_counts.get('tag', 0)}")
    console.print()

    # Show note node details
    nodes = graph.get("nodes", [])
    note_nodes = [n for n in nodes if n.get("type") == "note"]
    if note_nodes:
        note_table = Table(title="Note Nodes", show_lines=True)
        note_table.add_column("ID", style="cyan", max_width=35)
        note_table.add_column("Label", style="white", max_width=35)
        note_table.add_column("Group", style="green", max_width=14)
        note_table.add_column("Words", style="yellow", max_width=8)

        for node in note_nodes:
            note_table.add_row(
                node["id"][:35],
                node["label"][:35],
                node.get("group", ""),
                str(node.get("word_count", 0)),
            )
        console.print(note_table)
        console.print()


def cli_entrypoint():
    """Smart CLI entrypoint allowing direct execution or explicit subcommands."""
    _fix_windows_encoding()
    if len(sys.argv) == 1:
        sys.argv.insert(1, "run")
    elif len(sys.argv) > 1:
        first_arg = sys.argv[1]
        valid_commands = ["run", "status", "--help", "-h"]
        if first_arg not in valid_commands and not first_arg.startswith("-"):
            sys.argv.insert(1, "run")
    main()


if __name__ == "__main__":
    cli_entrypoint()
