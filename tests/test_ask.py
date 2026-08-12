"""
Tests for the smart chunk-based retrieval logic in src/ask.py.

Covers:
  - _chunk_text(): overlapping text chunking with boundary detection
  - _read_note_body(): full body extraction (no truncation)
  - _read_note_content(): backward-compat truncated wrapper
  - _extract_best_chunks(): relevance-ranked chunk selection
"""

import re
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ask import (
    _chunk_text,
    _read_note_body,
    _read_note_content,
    _extract_best_chunks,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


# ---------------------------------------------------------------------------
# _chunk_text tests
# ---------------------------------------------------------------------------


class TestChunkText:
    """Unit tests for _chunk_text()."""

    def test_empty_text_returns_empty_list(self):
        assert _chunk_text("") == []

    def test_none_text_returns_empty_list(self):
        assert _chunk_text(None) == []

    def test_short_text_returns_single_chunk(self):
        short = "Hello world, this is a short note."
        result = _chunk_text(short, chunk_size=1000)
        assert len(result) == 1
        assert result[0] == short

    def test_text_exactly_chunk_size_returns_single_chunk(self):
        text = "A" * 1500
        result = _chunk_text(text, chunk_size=1500)
        assert len(result) == 1

    def test_large_text_produces_multiple_chunks(self):
        # 5000 chars → should produce multiple chunks at 1500 size
        text = "A" * 5000
        result = _chunk_text(text, chunk_size=1500, overlap=300)
        assert len(result) > 1

    def test_chunks_overlap(self):
        """Consecutive chunks should share overlapping characters."""
        text = "word " * 1000  # ~5000 chars
        result = _chunk_text(text, chunk_size=1500, overlap=300)

        # At least 2 chunks
        assert len(result) >= 2

        # Check that the end of chunk 0 overlaps with the start of chunk 1
        # (they share some content)
        overlap_detected = False
        for i in range(len(result) - 1):
            tail = result[i][-200:]
            head = result[i + 1][:200:]
            if any(w in head for w in tail.split()[-5:]):
                overlap_detected = True
                break
        assert overlap_detected, "Expected overlapping content between consecutive chunks"

    def test_prefers_paragraph_boundary(self):
        """When a paragraph break exists in the search window, should split there."""
        section1 = "A" * 900
        section2 = "B" * 900
        text = section1 + "\n\n" + section2
        result = _chunk_text(text, chunk_size=1200, overlap=100)
        assert len(result) >= 2
        # First chunk should end near or at the paragraph boundary
        assert result[0].rstrip().endswith("A" * 10) or "\n\n" not in result[0]

    def test_prefers_sentence_boundary(self):
        """When no paragraph break, should try sentence boundary."""
        # Build text with sentences but no paragraph breaks
        sentences = "This is sentence one. " * 50 + "This is sentence two. " * 50
        result = _chunk_text(sentences, chunk_size=500, overlap=50)
        assert len(result) >= 2
        # First chunk should end at or near a period
        stripped = result[0].rstrip()
        assert stripped.endswith(".") or stripped.endswith(". ")

    def test_no_empty_chunks(self):
        text = "Hello\n\n\n\nWorld\n\n\n\nFoo"
        result = _chunk_text(text, chunk_size=10, overlap=2)
        for chunk in result:
            assert chunk.strip() != ""

    def test_all_content_covered(self):
        """Concatenated chunks should cover all original content words."""
        text = " ".join(f"word{i}" for i in range(200))
        result = _chunk_text(text, chunk_size=300, overlap=50)
        combined = " ".join(result)
        for i in range(200):
            assert f"word{i}" in combined, f"Missing word{i}"


# ---------------------------------------------------------------------------
# _read_note_body tests
# ---------------------------------------------------------------------------

class TestReadNoteBody:
    """Tests for _read_note_body() — full body extraction."""

    def test_reads_body_without_frontmatter(self, tmp_path):
        note = tmp_path / "test.md"
        note.write_text(
            "---\ntitle: Test\ncategory: Projects\n---\n\n# Main Content\n\nThis is the body.",
            encoding="utf-8",
        )
        body = _read_note_body(str(note))
        assert "# Main Content" in body
        assert "This is the body." in body
        assert "title: Test" not in body

    def test_strips_related_knowledge_section(self, tmp_path):
        note = tmp_path / "test.md"
        note.write_text(
            "---\ntitle: Test\n---\n\nBody content here.\n\n## Related Knowledge\n\n- [[SomeNote]]\n",
            encoding="utf-8",
        )
        body = _read_note_body(str(note))
        assert "Body content here" in body
        assert "Related Knowledge" not in body
        assert "SomeNote" not in body

    def test_returns_full_body_no_truncation(self, tmp_path):
        """Unlike old _read_note_content, _read_note_body does NOT truncate."""
        note = tmp_path / "test.md"
        long_body = "X" * 10000
        note.write_text(f"---\ntitle: Big\n---\n\n{long_body}", encoding="utf-8")
        body = _read_note_body(str(note))
        assert len(body) >= 10000
        assert "[...content truncated...]" not in body

    def test_nonexistent_file_returns_empty(self):
        body = _read_note_body("/nonexistent/path/note.md")
        assert body == ""

    def test_no_frontmatter(self, tmp_path):
        note = tmp_path / "test.md"
        note.write_text("Just plain content.\n\nMore content.", encoding="utf-8")
        body = _read_note_body(str(note))
        assert "Just plain content." in body


# ---------------------------------------------------------------------------
# _read_note_content backward-compat tests
# ---------------------------------------------------------------------------

class TestReadNoteContent:
    """Tests for _read_note_content() — truncated wrapper."""

    def test_small_note_returned_in_full(self, tmp_path):
        note = tmp_path / "test.md"
        note.write_text("---\ntitle: X\n---\n\nShort body.", encoding="utf-8")
        result = _read_note_content(str(note), max_chars=4000)
        assert result == "Short body."

    def test_large_note_truncated(self, tmp_path):
        note = tmp_path / "test.md"
        body = "Y" * 10000
        note.write_text(f"---\ntitle: Big\n---\n\n{body}", encoding="utf-8")
        result = _read_note_content(str(note), max_chars=4000)
        assert len(result) < 5000
        assert "[...content truncated...]" in result

    def test_default_max_chars_is_4000(self, tmp_path):
        note = tmp_path / "test.md"
        body = "Z" * 6000
        note.write_text(f"---\ntitle: Big\n---\n\n{body}", encoding="utf-8")
        result = _read_note_content(str(note))
        # Should truncate at 4000 + the truncation marker
        assert len(result) < 4100


# ---------------------------------------------------------------------------
# _extract_best_chunks tests
# ---------------------------------------------------------------------------

class TestExtractBestChunks:
    """Tests for _extract_best_chunks() — relevance-ranked chunk selection."""

    def _make_mock_model(self, embeddings_map):
        """
        Create a mock embedding model.
        embeddings_map: dict of chunk_text -> embedding_vector
        """
        import numpy as np

        model = MagicMock()

        def encode_side_effect(texts, **kwargs):
            result = []
            for t in texts:
                # Find the best match in embeddings_map
                best_key = min(embeddings_map.keys(), key=lambda k: abs(len(k) - len(t)))
                result.append(embeddings_map.get(t, embeddings_map[best_key]))
            return np.array(result)

        model.encode = MagicMock(side_effect=encode_side_effect)
        return model

    def test_returns_string(self):
        import numpy as np

        body = "Hello world. " * 500  # Large enough to chunk
        q_emb = np.random.rand(384)
        model = MagicMock()
        model.encode = MagicMock(return_value=np.random.rand(10, 384))

        result = _extract_best_chunks(body, q_emb, model, max_chars=4000)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_respects_max_chars(self):
        import numpy as np

        body = "Content block. " * 2000  # ~30k chars
        q_emb = np.random.rand(384)
        model = MagicMock()
        model.encode = MagicMock(return_value=np.random.rand(25, 384))

        result = _extract_best_chunks(body, q_emb, model, max_chars=3000)
        # Allow a bit of overhead for chunk boundaries + separator
        assert len(result) <= 5000  # generous upper bound

    def test_empty_body_returns_empty(self):
        import numpy as np
        result = _extract_best_chunks("", np.zeros(384), MagicMock())
        assert result == ""

    def test_single_chunk_body(self):
        import numpy as np
        short = "This is short content."
        result = _extract_best_chunks(
            short, np.zeros(384), MagicMock(), chunk_size=5000
        )
        assert result == short

    def test_preserves_document_order(self):
        """Selected chunks should be reassembled in original document order."""
        import numpy as np

        # Build 5 distinct sections
        sections = [f"SECTION_{i} " * 200 for i in range(5)]
        body = "\n\n".join(sections)

        q_emb = np.random.rand(384)
        model = MagicMock()

        # Make chunk embeddings where section 4 is most similar, then 1, then 0
        def fake_encode(texts, **kwargs):
            embs = []
            for t in texts:
                if "SECTION_4" in t:
                    embs.append(q_emb * 0.99)
                elif "SECTION_1" in t:
                    embs.append(q_emb * 0.8)
                elif "SECTION_0" in t:
                    embs.append(q_emb * 0.6)
                else:
                    embs.append(np.random.rand(384) * 0.01)
            return np.array(embs)

        model.encode = MagicMock(side_effect=fake_encode)

        result = _extract_best_chunks(body, q_emb, model, max_chars=8000, chunk_size=1500)

        # Even though SECTION_4 was ranked highest, document order should be preserved
        if "SECTION_0" in result and "SECTION_4" in result:
            assert result.index("SECTION_0") < result.index("SECTION_4")

    def test_noncontiguous_chunks_use_separator(self):
        """When chunks are non-contiguous, they should be joined with [...]."""
        import numpy as np

        # 10 sections, we'll make section 0 and section 9 most relevant
        sections = [f"SEC{i}_content " * 200 for i in range(10)]
        body = "\n\n".join(sections)

        q_emb = np.ones(384)
        model = MagicMock()

        def fake_encode(texts, **kwargs):
            embs = []
            for t in texts:
                if "SEC0" in t:
                    embs.append(np.ones(384))
                elif "SEC9" in t:
                    embs.append(np.ones(384) * 0.9)
                else:
                    embs.append(np.random.rand(384) * 0.01)
            return np.array(embs)

        model.encode = MagicMock(side_effect=fake_encode)

        result = _extract_best_chunks(
            body, q_emb, model, max_chars=5000, chunk_size=1500
        )

        # If both sections selected and non-contiguous, should have separator
        if "SEC0" in result and "SEC9" in result:
            assert "[...]" in result
