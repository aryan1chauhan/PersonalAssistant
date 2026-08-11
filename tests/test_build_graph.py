"""
Unit and integration tests for SecondSelf Knowledge Graph Data Model Exporter
(Phase 4.1 — The Cartographer).

Tests create wiki notes in tmp_path fixtures — no real wiki data or model
downloads required.
"""

import json
import re
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pytest

from src.build_graph import (
    build_note_nodes,
    build_category_nodes,
    build_tag_nodes,
    build_category_edges,
    build_tag_edges,
    build_wikilink_edges,
    build_semantic_edges,
    build_graph,
    export_graph_json,
    _note_slug,
    _extract_wikilink_titles,
)
from src.link import scan_wiki_notes, PARA_CATEGORIES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wiki_note(
    directory: Path,
    category: str,
    title: str,
    body: str,
    raw_id: str = "raw_test_001",
    summary: str = "Test summary",
    tags: Optional[List[str]] = None,
    related_titles: Optional[List[str]] = None,
) -> Path:
    """Create a wiki markdown note file with proper frontmatter."""
    tags = tags or ["test"]
    cat_dir = directory / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    filepath = cat_dir / f"{slug}.md"

    tags_yaml = "\n".join(f"  - {t}" for t in tags)
    content = f'''---
id: "{raw_id}"
title: "{title}"
category: "{category}"
tags:
{tags_yaml}
summary: "{summary}"
created_at: "2026-08-11T00:00:00+05:30"
source: "test"
type: "note"
classified_by: "rule"
---

# {title}

> **Summary**: {summary}

## Content

{body}
'''

    if related_titles:
        content += "\n## Related Knowledge\n\n"
        for rt in related_titles:
            content += f"- [[{rt}]]\n"

    filepath.write_text(content, encoding="utf-8")
    return filepath


def _make_embeddings_pkl(
    data_dir: Path,
    notes: List[Dict[str, Any]],
    embeddings: Optional[List[Any]] = None,
) -> Path:
    """Create a mock embeddings.pkl file for semantic edge testing."""
    data_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = data_dir / "embeddings.pkl"

    if embeddings is None:
        # Generate random embeddings
        embeddings = [np.random.randn(384).astype(np.float32) for _ in notes]

    records = []
    for i, note in enumerate(notes):
        records.append({
            "title": note["title"],
            "category": note["category"],
            "tags": note.get("tags", []),
            "summary": note.get("summary", ""),
            "path": str(note["path"]),
            "word_count": note["word_count"],
            "embedding": np.array(embeddings[i]),
        })

    payload = {
        "model": "all-MiniLM-L6-v2",
        "dimension": 384,
        "count": len(records),
        "notes": records,
    }

    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f)

    return pkl_path


# ---------------------------------------------------------------------------
# Tests: Node Builders
# ---------------------------------------------------------------------------

class TestBuildNoteNodes:
    """Tests for note node construction."""

    def test_build_nodes_from_notes(self, tmp_path):
        """Note nodes are created with correct schema from parsed wiki notes."""
        wiki_dir = tmp_path / "wiki"
        _make_wiki_note(wiki_dir, "1_Projects", "Alpha Project", "Building alpha.", tags=["ai", "ml"])
        _make_wiki_note(wiki_dir, "3_Resources", "Python Guide", "Learn Python.", tags=["python"])

        notes = scan_wiki_notes(wiki_dir)
        assert len(notes) == 2

        nodes = build_note_nodes(notes)
        assert len(nodes) == 2

        # Verify schema fields
        for node in nodes:
            assert "id" in node
            assert "label" in node
            assert "group" in node
            assert node["type"] == "note"
            assert "summary" in node
            assert "tags" in node
            assert "word_count" in node
            assert "file_path" in node

        # Verify specific values
        labels = {n["label"] for n in nodes}
        assert "Alpha Project" in labels
        assert "Python Guide" in labels

        groups = {n["group"] for n in nodes}
        assert "1_Projects" in groups
        assert "3_Resources" in groups

    def test_isolated_singleton_nodes(self, tmp_path):
        """Notes with no links still appear as nodes."""
        wiki_dir = tmp_path / "wiki"
        _make_wiki_note(wiki_dir, "4_Archives", "Old Note", "Archived content.")

        notes = scan_wiki_notes(wiki_dir)
        nodes = build_note_nodes(notes)
        assert len(nodes) == 1
        assert nodes[0]["label"] == "Old Note"
        assert nodes[0]["group"] == "4_Archives"

    def test_special_characters_in_titles(self, tmp_path):
        """Titles with ampersands, brackets, and special chars don't break node creation."""
        wiki_dir = tmp_path / "wiki"
        _make_wiki_note(
            wiki_dir, "3_Resources",
            "Note with Ampersand & [Brackets] + Dashes",
            "Content with special characters.",
        )

        notes = scan_wiki_notes(wiki_dir)
        nodes = build_note_nodes(notes)
        assert len(nodes) == 1
        assert "Ampersand" in nodes[0]["label"]
        assert nodes[0]["type"] == "note"


class TestBuildCategoryNodes:
    """Tests for category node construction."""

    def test_category_nodes_only_for_populated(self, tmp_path):
        """Category nodes are only created for categories that contain notes."""
        wiki_dir = tmp_path / "wiki"
        # Create notes in only 2 of 4 categories
        _make_wiki_note(wiki_dir, "1_Projects", "Proj A", "Content A.")
        _make_wiki_note(wiki_dir, "3_Resources", "Res B", "Content B.")
        # Ensure empty category dirs exist
        (wiki_dir / "2_Areas").mkdir(parents=True, exist_ok=True)
        (wiki_dir / "4_Archives").mkdir(parents=True, exist_ok=True)

        notes = scan_wiki_notes(wiki_dir)
        cat_nodes = build_category_nodes(notes)

        cat_ids = {n["id"] for n in cat_nodes}
        assert "category/1_Projects" in cat_ids
        assert "category/3_Resources" in cat_ids
        assert "category/2_Areas" not in cat_ids
        assert "category/4_Archives" not in cat_ids

        for node in cat_nodes:
            assert node["type"] == "category"


class TestBuildTagNodes:
    """Tests for tag node construction."""

    def test_duplicate_tag_deduplication(self, tmp_path):
        """Tags shared across multiple notes produce exactly one tag node."""
        wiki_dir = tmp_path / "wiki"
        _make_wiki_note(wiki_dir, "1_Projects", "Note A", "Content.", tags=["ai", "python"])
        _make_wiki_note(wiki_dir, "3_Resources", "Note B", "Content.", tags=["ai", "ml"])

        notes = scan_wiki_notes(wiki_dir)
        tag_nodes = build_tag_nodes(notes)

        tag_ids = [n["id"] for n in tag_nodes]
        # "ai" appears in both notes but should produce exactly one node
        assert tag_ids.count("tag/ai") == 1
        # All three unique tags should be present
        assert len(tag_nodes) == 3
        tag_labels = {n["label"] for n in tag_nodes}
        assert tag_labels == {"ai", "python", "ml"}

        for node in tag_nodes:
            assert node["type"] == "tag"
            assert node["group"] == "tags"


# ---------------------------------------------------------------------------
# Tests: Edge Builders
# ---------------------------------------------------------------------------

class TestBuildWikilinkEdges:
    """Tests for explicit [[wikilink]] edge construction."""

    def test_build_edges_wikilinks(self, tmp_path):
        """Explicit wikilink edges are extracted from ## Related Knowledge sections."""
        wiki_dir = tmp_path / "wiki"
        _make_wiki_note(
            wiki_dir, "1_Projects", "Note Alpha", "Alpha content is here for testing.",
            tags=["test"], related_titles=["Note Beta"],
        )
        _make_wiki_note(
            wiki_dir, "1_Projects", "Note Beta", "Beta content is here for testing.",
            tags=["test"], related_titles=["Note Alpha"],
        )

        notes = scan_wiki_notes(wiki_dir)
        edges = build_wikilink_edges(notes)

        # Bidirectional links in files, but should produce one deduplicated edge
        assert len(edges) == 1
        assert edges[0]["type"] == "explicit"
        assert edges[0]["weight"] == 1.0

        # Both slugs should be present across from/to
        slugs = {edges[0]["from"], edges[0]["to"]}
        assert len(slugs) == 2

    def test_dangling_wikilinks_skipped(self, tmp_path):
        """Wikilinks pointing to non-existent notes are silently skipped."""
        wiki_dir = tmp_path / "wiki"
        _make_wiki_note(
            wiki_dir, "1_Projects", "Existing Note", "Real note content here.",
            tags=["test"], related_titles=["Non Existent Note"],
        )

        notes = scan_wiki_notes(wiki_dir)
        edges = build_wikilink_edges(notes)
        assert len(edges) == 0

    def test_self_links_skipped(self, tmp_path):
        """Wikilinks pointing to the note itself are skipped."""
        wiki_dir = tmp_path / "wiki"
        _make_wiki_note(
            wiki_dir, "1_Projects", "Self Ref", "Content about self referencing.",
            tags=["test"], related_titles=["Self Ref"],
        )

        notes = scan_wiki_notes(wiki_dir)
        edges = build_wikilink_edges(notes)
        assert len(edges) == 0


class TestBuildSemanticEdges:
    """Tests for semantic similarity edges loaded from embeddings.pkl."""

    def test_build_edges_semantic(self, tmp_path):
        """Semantic edges are created from embeddings.pkl for similar notes."""
        wiki_dir = tmp_path / "wiki"
        data_dir = tmp_path / "data"

        _make_wiki_note(wiki_dir, "1_Projects", "AI Project", "Building AI systems and models.")
        _make_wiki_note(wiki_dir, "1_Projects", "ML Research", "Machine learning research and development.")

        notes = scan_wiki_notes(wiki_dir)

        # Create embeddings that are very similar (nearly identical vectors)
        base_vec = np.ones(384, dtype=np.float32)
        similar_vec = base_vec + np.random.randn(384).astype(np.float32) * 0.01
        _make_embeddings_pkl(data_dir, notes, [base_vec, similar_vec])

        edges = build_semantic_edges(notes, data_dir=data_dir, threshold=0.5)
        assert len(edges) >= 1
        assert edges[0]["type"] == "semantic"
        assert edges[0]["weight"] > 0.5

    def test_no_embeddings_file(self, tmp_path):
        """Semantic edges return empty list when no embeddings.pkl exists."""
        wiki_dir = tmp_path / "wiki"
        data_dir = tmp_path / "data"

        _make_wiki_note(wiki_dir, "1_Projects", "Note One", "Content one.")
        notes = scan_wiki_notes(wiki_dir)

        edges = build_semantic_edges(notes, data_dir=data_dir)
        assert edges == []


# ---------------------------------------------------------------------------
# Tests: Full Pipeline
# ---------------------------------------------------------------------------

class TestBuildGraph:
    """Tests for the complete graph building pipeline."""

    def test_empty_wiki(self, tmp_path):
        """Graceful handling of zero notes — returns empty graph."""
        wiki_dir = tmp_path / "wiki"
        for cat in PARA_CATEGORIES:
            (wiki_dir / cat).mkdir(parents=True, exist_ok=True)

        graph = build_graph(wiki_dir=wiki_dir, data_dir=tmp_path / "data")

        assert graph["metadata"]["total_nodes"] == 0
        assert graph["metadata"]["total_edges"] == 0
        assert graph["nodes"] == []
        assert graph["edges"] == []

    def test_export_graph_json(self, tmp_path):
        """Full pipeline writes valid graph.json with correct schema."""
        wiki_dir = tmp_path / "wiki"
        _make_wiki_note(wiki_dir, "1_Projects", "Project X", "Project X content for graph test.",
                        tags=["dev", "ai"], summary="Project X summary",
                        related_titles=["Resource Y"])
        _make_wiki_note(wiki_dir, "3_Resources", "Resource Y", "Resource Y reference material here.",
                        tags=["ai", "reference"], summary="Resource Y summary",
                        related_titles=["Project X"])

        graph = build_graph(wiki_dir=wiki_dir, data_dir=tmp_path / "data")

        # Export to JSON file
        out_path = tmp_path / "graph.json"
        result_path = export_graph_json(graph, output_path=out_path)
        assert result_path == out_path
        assert out_path.exists()

        # Validate JSON is parseable
        with open(out_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        # Validate top-level schema
        assert "metadata" in loaded
        assert "nodes" in loaded
        assert "edges" in loaded
        assert loaded["metadata"]["total_nodes"] == len(loaded["nodes"])
        assert loaded["metadata"]["total_edges"] == len(loaded["edges"])

        # Validate node types present
        node_types = {n["type"] for n in loaded["nodes"]}
        assert "note" in node_types
        assert "category" in node_types
        assert "tag" in node_types

        # Validate edge types present
        edge_types = {e["type"] for e in loaded["edges"]}
        assert "category" in edge_types
        assert "tag" in edge_types
        assert "explicit" in edge_types  # wikilink edges

    def test_full_pipeline_integration(self, tmp_path):
        """End-to-end: create wiki notes → build graph → validate structure."""
        wiki_dir = tmp_path / "wiki"
        data_dir = tmp_path / "data"

        # Create a small knowledge base
        _make_wiki_note(wiki_dir, "1_Projects", "SecondSelf Brain",
                        "Building a personal AI second brain with embeddings and RAG.",
                        tags=["ai", "brain", "project"],
                        related_titles=["ML Fundamentals"])
        _make_wiki_note(wiki_dir, "3_Resources", "ML Fundamentals",
                        "Machine learning fundamentals: supervised, unsupervised, reinforcement learning.",
                        tags=["ai", "ml", "learning"],
                        related_titles=["SecondSelf Brain"])
        _make_wiki_note(wiki_dir, "2_Areas", "Career Growth",
                        "Focus on AI engineering career development and skill building.",
                        tags=["career", "ai"])
        _make_wiki_note(wiki_dir, "4_Archives", "Old Project",
                        "This project was completed last quarter.",
                        tags=["archive"])

        graph = build_graph(wiki_dir=wiki_dir, data_dir=data_dir)

        # 4 note nodes + populated categories + unique tags
        meta = graph["metadata"]
        assert meta["node_counts"]["notes"] == 4

        # Categories: 1_Projects, 2_Areas, 3_Resources, 4_Archives (all populated)
        assert meta["node_counts"]["categories"] == 4

        # Tags: ai, brain, project, ml, learning, career, archive = 7
        assert meta["node_counts"]["tags"] == 7

        # Each note has category edge + tag edges
        assert meta["edge_counts"]["category"] == 4
        # Tag edges: 3 + 3 + 2 + 1 = 9
        assert meta["edge_counts"]["tag"] == 9

        # Wikilink edge: SecondSelf Brain <-> ML Fundamentals = 1 deduplicated
        assert meta["edge_counts"]["explicit"] == 1

        # No embeddings file, so no semantic edges
        assert meta["edge_counts"]["semantic"] == 0

        # Export and validate JSON roundtrip
        out_path = tmp_path / "graph.json"
        export_graph_json(graph, output_path=out_path)
        with open(out_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["metadata"]["total_nodes"] == meta["total_nodes"]
        assert loaded["metadata"]["total_edges"] == meta["total_edges"]

    def test_special_characters_in_json_export(self, tmp_path):
        """Titles with special characters produce valid JSON output."""
        wiki_dir = tmp_path / "wiki"
        _make_wiki_note(
            wiki_dir, "3_Resources",
            'Note with "Quotes" & Backslash\\Path',
            "Content testing JSON escaping of special characters.",
            tags=["special-chars"],
        )

        graph = build_graph(wiki_dir=wiki_dir, data_dir=tmp_path / "data")
        out_path = tmp_path / "graph.json"
        export_graph_json(graph, output_path=out_path)

        # Validate the JSON is parseable (would throw on broken escaping)
        with open(out_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        note_nodes = [n for n in loaded["nodes"] if n["type"] == "note"]
        assert len(note_nodes) == 1
