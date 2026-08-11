"""
Unit and integration tests for SecondSelf Dense Embeddings & Auto-Linking Engine
(Phase 2.2 — Connect the Dots).

Tests mock SentenceTransformer to avoid downloading the model during CI.
"""

import re
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from typing import Optional, List, Dict, Any

import numpy as np
import pytest

from src.link import (
    parse_wiki_note,
    scan_wiki_notes,
    compute_embeddings,
    cosine_similarity,
    compute_similarity_matrix,
    find_links,
    inject_wikilinks,
    clear_all_related_sections,
    run_auto_linking,
    _compose_embedding_text,
    _extract_existing_wikilinks,
    _strip_related_section,
    _build_related_section,
    PARA_CATEGORIES,
    RELATED_SECTION_HEADER,
    MIN_WORD_COUNT,
    MAX_LINKS_PER_NOTE,
)


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
) -> Path:
    """Create a wiki markdown note file with proper frontmatter."""
    tags = tags or ["test"]
    cat_dir = directory / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    filepath = cat_dir / f"{slug}.md"

    tags_yaml = "\n".join(f"  - {t}" for t in tags)
    content = (
        f'---\n'
        f'id: "{raw_id}"\n'
        f'title: "{title}"\n'
        f'category: "{category}"\n'
        f'tags:\n'
        f'{tags_yaml}\n'
        f'summary: "{summary}"\n'
        f'created_at: "2026-08-05T01:00:00+05:30"\n'
        f'source: "test"\n'
        f'type: "note"\n'
        f'classified_by: "rule"\n'
        f'---\n'
        f'\n'
        f'# {title}\n'
        f'\n'
        f'> **Summary**: {summary}\n'
        f'\n'
        f'## Content\n'
        f'\n'
        f'{body}\n'
    )
    filepath.write_text(content, encoding="utf-8")
    return filepath


def _make_fake_model():
    """Create a mock SentenceTransformer that returns deterministic embeddings."""
    model = MagicMock()

    def _encode(texts, **kwargs):
        """Generate deterministic embeddings based on text hash."""
        embeddings = []
        for text in texts:
            np.random.seed(hash(text) % (2**31))
            embeddings.append(np.random.randn(384).astype(np.float32))
        return np.array(embeddings)

    model.encode = _encode
    return model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def wiki_dir(tmp_path):
    """Create a temporary wiki directory with PARA category folders."""
    for cat in PARA_CATEGORIES:
        (tmp_path / cat).mkdir(parents=True)
    return tmp_path


@pytest.fixture
def sample_notes(wiki_dir):
    """Create a set of sample notes with varying content for testing."""
    notes_data = [
        {
            "category": "1_Projects",
            "title": "AI Project Development",
            "body": (
                "Building an AI-powered personal assistant using Python, embeddings, "
                "and large language models. The project includes vector search, "
                "PARA classification, and knowledge graph visualization."
            ),
            "raw_id": "raw_test_ai_001",
            "summary": "AI personal assistant project development",
            "tags": ["ai", "development", "project"],
        },
        {
            "category": "1_Projects",
            "title": "Machine Learning Pipeline",
            "body": (
                "Developing a machine learning pipeline for natural language processing. "
                "Using sentence transformers for embedding computation, cosine similarity "
                "for document matching, and vector databases for retrieval."
            ),
            "raw_id": "raw_test_ml_002",
            "summary": "ML pipeline for NLP tasks",
            "tags": ["ml", "nlp", "pipeline"],
        },
        {
            "category": "3_Resources",
            "title": "Cooking Recipes Collection",
            "body": (
                "A collection of favorite cooking recipes including Italian pasta dishes, "
                "Japanese ramen bowls, Mexican tacos, and Indian curry variations. "
                "Each recipe includes ingredient lists and step-by-step instructions."
            ),
            "raw_id": "raw_test_cook_003",
            "summary": "Favorite cooking recipes",
            "tags": ["cooking", "recipes", "food"],
        },
        {
            "category": "2_Areas",
            "title": "Gym Fitness Routine",
            "body": (
                "Weekly gym workout routine focusing on strength training. "
                "Monday: chest and triceps. Wednesday: back and biceps. "
                "Friday: legs and shoulders. Cardio on rest days."
            ),
            "raw_id": "raw_test_gym_004",
            "summary": "Weekly gym workout plan",
            "tags": ["fitness", "gym", "health"],
        },
    ]

    paths = []
    for data in notes_data:
        path = _make_wiki_note(
            directory=wiki_dir,
            category=str(data["category"]),
            title=str(data["title"]),
            body=str(data["body"]),
            raw_id=str(data["raw_id"]),
            summary=str(data["summary"]),
            tags=list(data["tags"]),
        )
        paths.append(path)

    return paths


@pytest.fixture
def short_note(wiki_dir):
    """Create a note with fewer than MIN_WORD_COUNT words."""
    return _make_wiki_note(
        wiki_dir,
        category="4_Archives",
        title="Short Note",
        body="Too short.",
        raw_id="raw_test_short",
        summary="Very short note",
    )


# ---------------------------------------------------------------------------
# Wiki Note Parser Tests
# ---------------------------------------------------------------------------

class TestParseWikiNote:
    def test_parses_valid_note(self, sample_notes):
        parsed = parse_wiki_note(sample_notes[0])
        assert parsed is not None
        assert parsed["title"] == "AI Project Development"
        assert parsed["category"] == "1_Projects"
        assert "ai" in parsed["tags"]
        assert parsed["word_count"] > 0
        assert "AI-powered" in parsed["body"]

    def test_extracts_summary(self, sample_notes):
        parsed = parse_wiki_note(sample_notes[0])
        assert parsed["summary"] == "AI personal assistant project development"

    def test_nonexistent_file_returns_none(self, tmp_path):
        result = parse_wiki_note(tmp_path / "nonexistent.md")
        assert result is None

    def test_no_frontmatter_returns_none(self, tmp_path):
        bad_file = tmp_path / "no_fm.md"
        bad_file.write_text("# Just a heading\nNo frontmatter here.", encoding="utf-8")
        assert parse_wiki_note(bad_file) is None

    def test_word_count_calculation(self, wiki_dir):
        path = _make_wiki_note(
            wiki_dir,
            "3_Resources",
            "Word Count Test",
            "one two three four five six seven eight nine ten",
            raw_id="raw_wc_test",
        )
        parsed = parse_wiki_note(path)
        assert parsed["word_count"] >= 10

    def test_related_section_excluded_from_body(self, wiki_dir):
        """The ## Related Knowledge section should NOT be included in body for embedding."""
        path = _make_wiki_note(
            wiki_dir,
            "3_Resources",
            "Linked Note",
            "Main content about science and research.",
            raw_id="raw_linked_test",
        )
        # Append a related section
        content = path.read_text(encoding="utf-8")
        content += "\n## Related Knowledge\n\n- [[Some Other Note]]\n"
        path.write_text(content, encoding="utf-8")

        parsed = parse_wiki_note(path)
        assert "Some Other Note" not in parsed["body"]
        assert "science" in parsed["body"]


class TestScanWikiNotes:
    def test_finds_all_notes(self, wiki_dir, sample_notes):
        notes = scan_wiki_notes(wiki_dir)
        assert len(notes) == 4

    def test_empty_wiki(self, wiki_dir):
        notes = scan_wiki_notes(wiki_dir)
        assert len(notes) == 0

    def test_skips_gitkeep(self, wiki_dir):
        (wiki_dir / "1_Projects" / ".gitkeep").write_text("", encoding="utf-8")
        notes = scan_wiki_notes(wiki_dir)
        assert len(notes) == 0


# ---------------------------------------------------------------------------
# Embedding Text Composition Tests
# ---------------------------------------------------------------------------

class TestComposeEmbeddingText:
    def test_combines_title_summary_body(self):
        note = {
            "title": "My Title",
            "summary": "Brief summary",
            "body": "Full body content here.",
        }
        text = _compose_embedding_text(note)
        assert "My Title" in text
        assert "Brief summary" in text
        assert "Full body content here" in text

    def test_handles_empty_fields(self):
        note = {"title": "", "summary": "", "body": "Only body."}
        text = _compose_embedding_text(note)
        assert "Only body" in text


# ---------------------------------------------------------------------------
# Cosine Similarity Tests
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        vec = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(vec, vec) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_zero_vector(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(a, b) == 0.0

    def test_similar_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.1, 2.1, 3.1])
        sim = cosine_similarity(a, b)
        assert sim > 0.99  # Very similar


class TestComputeSimilarityMatrix:
    def test_diagonal_is_one(self):
        embeddings = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        matrix = compute_similarity_matrix(embeddings)
        for i in range(3):
            assert matrix[i][i] == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_off_diagonal(self):
        embeddings = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        ]
        matrix = compute_similarity_matrix(embeddings)
        assert matrix[0][1] == pytest.approx(0.0, abs=1e-6)
        assert matrix[1][0] == pytest.approx(0.0, abs=1e-6)

    def test_symmetric_matrix(self):
        embeddings = [np.random.randn(10) for _ in range(5)]
        matrix = compute_similarity_matrix(embeddings)
        for i in range(5):
            for j in range(5):
                assert matrix[i][j] == pytest.approx(matrix[j][i], abs=1e-6)

    def test_empty_embeddings(self):
        matrix = compute_similarity_matrix([])
        assert matrix == []


# ---------------------------------------------------------------------------
# Link Finding Tests
# ---------------------------------------------------------------------------

class TestFindLinks:
    def _make_notes_with_word_counts(self, counts):
        return [{"word_count": c, "title": f"Note {i}"} for i, c in enumerate(counts)]

    def test_self_links_excluded(self):
        notes = self._make_notes_with_word_counts([20, 20])
        sim_matrix = [[1.0, 0.9], [0.9, 1.0]]
        links = find_links(notes, sim_matrix, threshold=0.5)
        for i, targets in links.items():
            for j, _ in targets:
                assert i != j

    def test_threshold_filtering(self):
        notes = self._make_notes_with_word_counts([20, 20, 20])
        sim_matrix = [
            [1.0, 0.8, 0.3],
            [0.8, 1.0, 0.4],
            [0.3, 0.4, 1.0],
        ]
        links = find_links(notes, sim_matrix, threshold=0.65)
        # Note 0 and 1 should be linked (0.8 >= 0.65)
        assert any(j == 1 for j, _ in links[0])
        # Note 0 and 2 should NOT be linked (0.3 < 0.65)
        assert not any(j == 2 for j, _ in links[0])

    def test_short_notes_skipped(self):
        notes = self._make_notes_with_word_counts([20, 5, 20])  # Note 1 is too short
        sim_matrix = [
            [1.0, 0.9, 0.8],
            [0.9, 1.0, 0.9],
            [0.8, 0.9, 1.0],
        ]
        links = find_links(notes, sim_matrix, threshold=0.5, min_words=MIN_WORD_COUNT)
        # Note 1 (short) should have no links
        assert len(links[1]) == 0
        # Note 0 should not link to note 1 (short target)
        assert not any(j == 1 for j, _ in links[0])

    def test_max_links_cap(self):
        n = 10
        notes = self._make_notes_with_word_counts([20] * n)
        # All pairs have high similarity
        sim_matrix = [[0.9 if i != j else 1.0 for j in range(n)] for i in range(n)]
        links = find_links(notes, sim_matrix, threshold=0.5, max_links=3)
        for targets in links.values():
            assert len(targets) <= 3

    def test_sorted_by_similarity_descending(self):
        notes = self._make_notes_with_word_counts([20, 20, 20, 20])
        sim_matrix = [
            [1.0, 0.7, 0.9, 0.8],
            [0.7, 1.0, 0.7, 0.7],
            [0.9, 0.7, 1.0, 0.7],
            [0.8, 0.7, 0.7, 1.0],
        ]
        links = find_links(notes, sim_matrix, threshold=0.65, max_links=5)
        # Note 0's links should be sorted: note 2 (0.9), note 3 (0.8), note 1 (0.7)
        scores = [score for _, score in links[0]]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Wikilink Extraction & Section Management Tests
# ---------------------------------------------------------------------------

class TestWikilinkHelpers:
    def test_extract_existing_wikilinks(self):
        text = "Some text with [[Note A]] and [[Note B]] inside."
        links = _extract_existing_wikilinks(text)
        assert links == {"Note A", "Note B"}

    def test_extract_no_wikilinks(self):
        text = "No wikilinks here."
        assert _extract_existing_wikilinks(text) == set()

    def test_strip_related_section(self):
        text = "# Title\n\nBody content.\n\n## Related Knowledge\n\n- [[Note A]]\n- [[Note B]]\n"
        result = _strip_related_section(text)
        assert "Related Knowledge" not in result
        assert "Note A" not in result
        assert "Body content" in result

    def test_strip_preserves_body(self):
        text = "# Title\n\nBody content.\n"
        result = _strip_related_section(text)
        assert "Body content" in result

    def test_build_related_section(self):
        section = _build_related_section(["Note A", "Note B"])
        assert "## Related Knowledge" in section
        assert "- [[Note A]]" in section
        assert "- [[Note B]]" in section


# ---------------------------------------------------------------------------
# Wikilink Injection Tests
# ---------------------------------------------------------------------------

class TestInjectWikilinks:
    def test_injects_bidirectional_links(self, wiki_dir):
        path_a = _make_wiki_note(
            wiki_dir, "1_Projects", "Note Alpha",
            "This is a long note about artificial intelligence and machine learning projects.",
            raw_id="raw_alpha",
        )
        path_b = _make_wiki_note(
            wiki_dir, "1_Projects", "Note Beta",
            "This is another long note about deep learning and neural network architectures.",
            raw_id="raw_beta",
        )

        notes = [parse_wiki_note(path_a), parse_wiki_note(path_b)]
        links = {0: [(1, 0.85)], 1: []}  # Only note 0 explicitly links to note 1

        modified = inject_wikilinks(notes, links)
        assert modified == 2  # Both should be modified (bidirectional)

        content_a = path_a.read_text(encoding="utf-8")
        content_b = path_b.read_text(encoding="utf-8")

        assert "[[Note Beta]]" in content_a
        assert "[[Note Alpha]]" in content_b

    def test_no_duplicate_links(self, wiki_dir):
        path_a = _make_wiki_note(
            wiki_dir, "3_Resources", "Existing Links Note",
            "This is a note that already has a wikilink to [[Already Linked]] in its body content area.",
            raw_id="raw_existing",
        )
        path_b = _make_wiki_note(
            wiki_dir, "3_Resources", "Already Linked",
            "This note is already linked from the other note about existing references and content.",
            raw_id="raw_already",
        )

        notes = [parse_wiki_note(path_a), parse_wiki_note(path_b)]
        links = {0: [(1, 0.9)], 1: []}

        inject_wikilinks(notes, links)

        content_a = path_a.read_text(encoding="utf-8")
        # The Related Knowledge section should NOT add "Already Linked" again
        related_section = content_a.split(RELATED_SECTION_HEADER)
        if len(related_section) > 1:
            assert content_a.count("[[Already Linked]]") == 1  # Only the original one

    def test_no_links_no_modification(self, wiki_dir):
        path = _make_wiki_note(
            wiki_dir, "3_Resources", "Lonely Note",
            "This note has no related notes at all and should remain untouched by the linker.",
            raw_id="raw_lonely",
        )
        notes = [parse_wiki_note(path)]
        links = {0: []}

        modified = inject_wikilinks(notes, links)
        assert modified == 0


# ---------------------------------------------------------------------------
# Clear Related Sections Tests
# ---------------------------------------------------------------------------

class TestClearRelatedSections:
    def test_clears_existing_sections(self, wiki_dir):
        path = _make_wiki_note(
            wiki_dir, "1_Projects", "Clearable Note",
            "Content here with words and sentences and paragraphs and more content.",
            raw_id="raw_clear",
        )
        # Add a related section
        content = path.read_text(encoding="utf-8")
        content += "\n## Related Knowledge\n\n- [[Some Note]]\n"
        path.write_text(content, encoding="utf-8")

        count = clear_all_related_sections(wiki_dir)
        assert count == 1

        cleaned = path.read_text(encoding="utf-8")
        assert RELATED_SECTION_HEADER not in cleaned

    def test_no_sections_to_clear(self, wiki_dir):
        _make_wiki_note(
            wiki_dir, "1_Projects", "No Links Note",
            "A plain note without any related knowledge section at all.",
            raw_id="raw_no_links",
        )
        count = clear_all_related_sections(wiki_dir)
        assert count == 0


# ---------------------------------------------------------------------------
# Compute Embeddings Tests (Mocked Model)
# ---------------------------------------------------------------------------

class TestComputeEmbeddings:
    def test_returns_correct_count(self, wiki_dir, sample_notes):
        notes = scan_wiki_notes(wiki_dir)
        model = _make_fake_model()
        embeddings = compute_embeddings(notes, model=model)
        assert len(embeddings) == len(notes)

    def test_embedding_dimensions(self, wiki_dir, sample_notes):
        notes = scan_wiki_notes(wiki_dir)
        model = _make_fake_model()
        embeddings = compute_embeddings(notes, model=model)
        for emb in embeddings:
            assert len(emb) == 384


# ---------------------------------------------------------------------------
# Full Pipeline Integration Test (Mocked Model)
# ---------------------------------------------------------------------------

class TestRunAutoLinking:
    def test_end_to_end_pipeline(self, wiki_dir):
        """Create related notes, run pipeline with mock model, verify links."""
        # Create two very similar notes and one different note
        _make_wiki_note(
            wiki_dir, "1_Projects", "Python Development Guide",
            (
                "A comprehensive guide to Python development covering best practices, "
                "design patterns, testing strategies, and code organization for "
                "building production-ready Python applications and libraries."
            ),
            raw_id="raw_py_001",
            summary="Python development best practices",
            tags=["python", "development"],
        )
        _make_wiki_note(
            wiki_dir, "3_Resources", "Python Testing Handbook",
            (
                "Handbook for Python testing including pytest fixtures, mocking, "
                "test-driven development workflow, and continuous integration "
                "setup for Python projects and application codebases."
            ),
            raw_id="raw_py_002",
            summary="Python testing and CI guide",
            tags=["python", "testing"],
        )
        _make_wiki_note(
            wiki_dir, "3_Resources", "Italian Cooking Guide",
            (
                "Traditional Italian cooking techniques for making fresh pasta, "
                "pizza dough, risotto, and classic sauces like bolognese and "
                "carbonara with authentic Italian ingredients and methods."
            ),
            raw_id="raw_cook_003",
            summary="Italian cooking techniques",
            tags=["cooking", "italian"],
        )

        # Create a mock model that returns similar embeddings for Python notes
        # and different embeddings for the cooking note
        model = MagicMock()
        python_vec = np.ones(384, dtype=np.float32)
        cooking_vec = -np.ones(384, dtype=np.float32)
        slightly_different = python_vec.copy()
        slightly_different[:10] = 0.5

        def _encode(texts, **kwargs):
            results = []
            for text in texts:
                if "python" in text.lower() or "testing" in text.lower():
                    results.append(slightly_different if "testing" in text.lower() else python_vec)
                else:
                    results.append(cooking_vec)
            return np.array(results)

        model.encode = _encode

        result = run_auto_linking(wiki_dir=wiki_dir, model=model, threshold=0.65)

        assert result["total_notes"] == 3
        assert result["total_links"] > 0

        # Verify the Python notes are linked to each other
        py_dev = (wiki_dir / "1_Projects" / "python_development_guide.md").read_text(encoding="utf-8")
        py_test = (wiki_dir / "3_Resources" / "python_testing_handbook.md").read_text(encoding="utf-8")
        cooking = (wiki_dir / "3_Resources" / "italian_cooking_guide.md").read_text(encoding="utf-8")

        assert "[[Python Testing Handbook]]" in py_dev
        assert "[[Python Development Guide]]" in py_test
        # Cooking note should NOT be linked to Python notes
        assert "[[Italian Cooking Guide]]" not in py_dev

    def test_empty_wiki(self, wiki_dir):
        result = run_auto_linking(wiki_dir=wiki_dir, model=_make_fake_model())
        assert result["total_notes"] == 0

    def test_idempotent_relinking(self, wiki_dir):
        """Running the pipeline twice should produce the same result."""
        _make_wiki_note(
            wiki_dir, "1_Projects", "Note One",
            "Machine learning and artificial intelligence research projects and applications.",
            raw_id="raw_one",
        )
        _make_wiki_note(
            wiki_dir, "1_Projects", "Note Two",
            "Deep learning neural network training and artificial intelligence deployment systems.",
            raw_id="raw_two",
        )

        model = MagicMock()
        vec1 = np.ones(384, dtype=np.float32)
        vec2 = vec1.copy()
        vec2[:5] = 0.9

        def _encode(texts, **kwargs):
            return np.array([vec1 if i == 0 else vec2 for i, _ in enumerate(texts)])

        model.encode = _encode

        # Run twice
        run_auto_linking(wiki_dir=wiki_dir, model=model, threshold=0.5)
        run_auto_linking(wiki_dir=wiki_dir, model=model, threshold=0.5)

        content = (wiki_dir / "1_Projects" / "note_one.md").read_text(encoding="utf-8")
        # Should only have ONE Related Knowledge section
        assert content.count(RELATED_SECTION_HEADER) == 1

    def test_short_notes_excluded(self, wiki_dir):
        """Notes below MIN_WORD_COUNT should not get or give links."""
        _make_wiki_note(
            wiki_dir, "1_Projects", "Long Note",
            "This is a sufficiently long note about technology and programming and software development.",
            raw_id="raw_long",
        )
        _make_wiki_note(
            wiki_dir, "4_Archives", "Tiny",
            "Short.",
            raw_id="raw_tiny",
        )

        model = MagicMock()
        vec = np.ones(384, dtype=np.float32)
        model.encode = lambda texts, **kwargs: np.array([vec] * len(texts))

        result = run_auto_linking(wiki_dir=wiki_dir, model=model, threshold=0.5)

        tiny_content = (wiki_dir / "4_Archives" / "tiny.md").read_text(encoding="utf-8")
        assert RELATED_SECTION_HEADER not in tiny_content
