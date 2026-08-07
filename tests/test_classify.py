"""
Unit and integration tests for SecondSelf AI Classification Engine (Phase 2.1 — The Sorting Hat).
"""

import json
import re
from pathlib import Path

import pytest

from src.classify import (
    classify_raw_item,
    write_wiki_note,
    batch_classify,
    _rule_based_classify,
    _extract_json,
    _escape_yaml_string,
    _get_classified_ids,
    PARA_CATEGORIES,
)
from src.utils import slugify


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_note_record():
    """A minimal note-type raw record."""
    return {
        "id": "raw_20260805_012359_test1234",
        "timestamp": "2026-08-05T01:23:59.000000+05:30",
        "type": "note",
        "source": "Building a Second Brain - Core CODE Method summary",
        "title": "CODE Framework Summary",
        "raw_content": (
            "Building a Second Brain - Core CODE Method: Capture (keep what resonates), "
            "Organize (for actionability via PARA), Distill (find the essence), "
            "Express (show your work). Information is useful only when it powers future action."
        ),
        "tags": ["uncategorized", "raw_ingest"],
    }


@pytest.fixture
def sample_link_record():
    """A link-type raw record."""
    return {
        "id": "raw_20260805_012417_linktest",
        "timestamp": "2026-08-05T01:24:17.000000+05:30",
        "type": "link",
        "source": "https://en.wikipedia.org/wiki/Getting_Things_Done",
        "title": "Getting Things Done (GTD) Methodology",
        "raw_content": (
            "Getting Things Done (GTD) is a personal productivity system developed by David Allen. "
            "The GTD method rests on the idea of moving all items of interest out of one's mind "
            "by recording them externally and then breaking them into actionable work items."
        ),
        "tags": ["uncategorized", "raw_ingest"],
        "scrape_status": "success",
    }


@pytest.fixture
def sample_file_record():
    """A file-type raw record with attachment metadata."""
    return {
        "id": "raw_20260805_013153_filetest",
        "timestamp": "2026-08-05T01:31:53.000000+05:30",
        "type": "file",
        "source": "sample_document.txt",
        "title": "File: sample_document.txt",
        "raw_content": (
            "# Sample Document for SecondSelf\n\n"
            "This is a real local text file captured into SecondSelf.\n"
            "It contains notes on System Architecture, PARA Categorization, and Vector Search."
        ),
        "tags": ["uncategorized", "raw_ingest"],
        "attachment": {
            "original_filename": "sample_document.txt",
            "stored_path": "raw/assets/sample.txt",
            "absolute_stored_path": "D:/PersonalAssistant/raw/assets/sample.txt",
            "size_bytes": 173,
            "sha256": "bf843dde",
            "extension": ".txt",
        },
    }


@pytest.fixture
def sample_project_record():
    """A record that should be classified as a Project by the rule-based classifier."""
    return {
        "id": "raw_20260805_000000_projtest",
        "timestamp": "2026-08-05T00:00:00.000000+05:30",
        "type": "note",
        "source": "Sprint planning notes",
        "title": "Q3 App Launch Sprint Plan",
        "raw_content": (
            "Sprint 5 deadline is August 15. Milestone: deploy MVP to production. "
            "Build the notification service and ship the release by end of sprint."
        ),
        "tags": ["uncategorized"],
    }


@pytest.fixture
def sample_archive_record():
    """A record that should be classified as Archive by the rule-based classifier."""
    return {
        "id": "raw_20260805_000000_archtest",
        "timestamp": "2026-08-05T00:00:00.000000+05:30",
        "type": "note",
        "source": "Old project wrap-up",
        "title": "Deprecated Legacy API Docs",
        "raw_content": (
            "This old legacy API is now deprecated and inactive. "
            "The project was completed and retired last quarter."
        ),
        "tags": ["uncategorized"],
    }


# ---------------------------------------------------------------------------
# Rule-Based Classifier Tests
# ---------------------------------------------------------------------------

class TestRuleBasedClassifier:
    def test_resource_classification(self, sample_note_record):
        """General knowledge content should fall into Resources."""
        result = _rule_based_classify(sample_note_record)
        assert result["category"] in PARA_CATEGORIES
        assert isinstance(result["tags"], list)
        assert len(result["tags"]) >= 1
        assert result["summary"]
        assert result["title"]

    def test_project_classification(self, sample_project_record):
        """Records with project keywords should be classified as Projects."""
        result = _rule_based_classify(sample_project_record)
        assert result["category"] == "1_Projects"

    def test_archive_classification(self, sample_archive_record):
        """Records with archive keywords should be classified as Archives."""
        result = _rule_based_classify(sample_archive_record)
        assert result["category"] == "4_Archives"

    def test_tags_are_lowercase_strings(self, sample_note_record):
        result = _rule_based_classify(sample_note_record)
        for tag in result["tags"]:
            assert isinstance(tag, str)
            assert tag == tag.lower()

    def test_empty_content_handled(self):
        """Should not crash on empty content."""
        record = {
            "id": "raw_empty",
            "type": "note",
            "source": "",
            "title": "",
            "raw_content": "",
        }
        result = _rule_based_classify(record)
        assert result["category"] in PARA_CATEGORIES
        assert result["title"]  # Should at least be "Untitled"


# ---------------------------------------------------------------------------
# classify_raw_item Tests (using rule-based provider)
# ---------------------------------------------------------------------------

class TestClassifyRawItem:
    def test_rule_provider_returns_valid_result(self, sample_note_record):
        result = classify_raw_item(sample_note_record, provider="rule")
        assert result["category"] in PARA_CATEGORIES
        assert isinstance(result["tags"], list)
        assert 1 <= len(result["tags"]) <= 5
        assert result["summary"]
        assert result["title"]
        assert result["provider"] == "rule"

    def test_invalid_provider_raises(self, sample_note_record):
        with pytest.raises(ValueError, match="Unknown provider"):
            classify_raw_item(sample_note_record, provider="nonexistent")

    def test_tags_normalized(self, sample_note_record):
        """Tags should be lowercase, hyphenated, with no spaces."""
        result = classify_raw_item(sample_note_record, provider="rule")
        for tag in result["tags"]:
            assert " " not in tag
            assert tag == tag.lower()


# ---------------------------------------------------------------------------
# JSON Extraction Tests
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_plain_json(self):
        raw = '{"category": "3_Resources", "tags": ["ai"], "summary": "test", "title": "Test"}'
        result = _extract_json(raw)
        assert result["category"] == "3_Resources"

    def test_json_with_markdown_fences(self):
        raw = '```json\n{"category": "1_Projects", "tags": ["dev"], "summary": "s", "title": "T"}\n```'
        result = _extract_json(raw)
        assert result["category"] == "1_Projects"

    def test_json_with_preamble(self):
        raw = 'Here is the classification:\n{"category": "2_Areas", "tags": ["health"], "summary": "s", "title": "T"}'
        result = _extract_json(raw)
        assert result["category"] == "2_Areas"

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Could not extract valid JSON"):
            _extract_json("This is not JSON at all")


# ---------------------------------------------------------------------------
# YAML Escaping Tests
# ---------------------------------------------------------------------------

class TestYamlEscape:
    def test_escape_quotes(self):
        assert _escape_yaml_string('Hello "world"') == 'Hello \\"world\\"'

    def test_escape_backslash(self):
        assert _escape_yaml_string("C:\\path\\to") == "C:\\\\path\\\\to"

    def test_strip_newlines(self):
        assert _escape_yaml_string("line1\nline2") == "line1 line2"


# ---------------------------------------------------------------------------
# Slug Generation Tests
# ---------------------------------------------------------------------------

class TestSlugGeneration:
    def test_basic_slug(self):
        assert slugify("Hello World!") == "hello_world"

    def test_special_chars_removed(self):
        s = slugify('My Note: A "Special" Case (v2.0)')
        assert '"' not in s
        assert ":" not in s
        assert "(" not in s

    def test_max_length_respected(self):
        s = slugify("A" * 100, max_length=50)
        assert len(s) <= 50

    def test_empty_input(self):
        assert slugify("") == "untitled"
        assert slugify("   ") == "untitled"


# ---------------------------------------------------------------------------
# Wiki Note Writer Tests
# ---------------------------------------------------------------------------

class TestWriteWikiNote:
    def test_creates_markdown_file(self, tmp_path, sample_note_record):
        classification = {
            "category": "3_Resources",
            "tags": ["productivity", "note-taking", "para"],
            "summary": "CODE framework for building a second brain",
            "title": "CODE Framework Summary",
            "provider": "rule",
        }
        path = write_wiki_note(sample_note_record, classification, wiki_dir=tmp_path)
        assert path.exists()
        assert path.suffix == ".md"
        assert "3_Resources" in str(path)

    def test_frontmatter_has_required_fields(self, tmp_path, sample_note_record):
        classification = {
            "category": "3_Resources",
            "tags": ["test"],
            "summary": "Test summary",
            "title": "Test Title",
            "provider": "rule",
        }
        path = write_wiki_note(sample_note_record, classification, wiki_dir=tmp_path)
        content = path.read_text(encoding="utf-8")

        assert content.startswith("---")
        assert "id:" in content
        assert "title:" in content
        assert "category:" in content
        assert "tags:" in content
        assert "summary:" in content
        assert "created_at:" in content
        assert "source:" in content
        assert "type:" in content
        assert "classified_by:" in content

    def test_link_type_includes_source_url(self, tmp_path, sample_link_record):
        classification = {
            "category": "3_Resources",
            "tags": ["productivity"],
            "summary": "GTD methodology overview",
            "title": "Getting Things Done",
            "provider": "rule",
        }
        path = write_wiki_note(sample_link_record, classification, wiki_dir=tmp_path)
        content = path.read_text(encoding="utf-8")
        assert "Source URL" in content
        assert "https://en.wikipedia.org" in content

    def test_file_type_includes_attachment(self, tmp_path, sample_file_record):
        classification = {
            "category": "3_Resources",
            "tags": ["architecture"],
            "summary": "Sample document with architecture notes",
            "title": "Sample Document",
            "provider": "rule",
        }
        path = write_wiki_note(sample_file_record, classification, wiki_dir=tmp_path)
        content = path.read_text(encoding="utf-8")
        assert "attachment_file:" in content
        assert "Original File" in content

    def test_slug_collision_resolved(self, tmp_path, sample_note_record):
        classification = {
            "category": "3_Resources",
            "tags": ["test"],
            "summary": "Test",
            "title": "Same Title",
            "provider": "rule",
        }
        path1 = write_wiki_note(sample_note_record, classification, wiki_dir=tmp_path)
        path2 = write_wiki_note(sample_note_record, classification, wiki_dir=tmp_path)
        assert path1 != path2
        assert path1.exists()
        assert path2.exists()


# ---------------------------------------------------------------------------
# Batch Classification Integration Test
# ---------------------------------------------------------------------------

class TestBatchClassify:
    def test_batch_e2e_with_rule_provider(self, tmp_path):
        """End-to-end: write sample raw JSONs → batch classify → verify wiki output."""
        raw_dir = tmp_path / "raw"
        wiki_dir = tmp_path / "wiki"
        raw_dir.mkdir()

        # Create 3 sample raw captures
        records = [
            {
                "id": "raw_test_note_001",
                "timestamp": "2026-08-05T01:00:00+05:30",
                "type": "note",
                "source": "Deadline for sprint 3 is next Friday. Must deploy MVP.",
                "title": "Sprint 3 Deadline Reminder",
                "raw_content": "Deadline for sprint 3 is next Friday. Must ship the MVP build and deploy to production.",
                "tags": ["uncategorized"],
            },
            {
                "id": "raw_test_note_002",
                "timestamp": "2026-08-05T02:00:00+05:30",
                "type": "note",
                "source": "Article on productivity methods",
                "title": "Pomodoro Technique Overview",
                "raw_content": "The Pomodoro Technique is a time management method using 25-minute focused intervals separated by 5-minute breaks.",
                "tags": ["uncategorized"],
            },
            {
                "id": "raw_test_note_003",
                "timestamp": "2026-08-05T03:00:00+05:30",
                "type": "note",
                "source": "Old unused reference",
                "title": "Deprecated API v1 Notes",
                "raw_content": "These old legacy API notes are now deprecated, obsolete, and inactive. The project was retired.",
                "tags": ["uncategorized"],
            },
        ]

        for rec in records:
            with open(raw_dir / f"{rec['id']}.json", "w", encoding="utf-8") as f:
                json.dump(rec, f)

        # Run batch classification with rule-based provider
        created = batch_classify(raw_dir=raw_dir, wiki_dir=wiki_dir, provider="rule")

        assert len(created) == 3

        # Verify files exist and have valid frontmatter
        for path in created:
            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert content.startswith("---")
            assert "category:" in content
            assert "tags:" in content

        # Verify PARA distribution — first record should be Project, third should be Archive
        all_content = {p.stem: p.read_text(encoding="utf-8") for p in created}

        # At least one file in each of the expected categories
        categories_found = set()
        for content in all_content.values():
            match = re.search(r'category:\s*"([^"]+)"', content)
            if match:
                categories_found.add(match.group(1))

        assert "1_Projects" in categories_found, "Sprint deadline record should be classified as Project"
        assert "4_Archives" in categories_found, "Deprecated API record should be classified as Archive"

    def test_skip_already_classified(self, tmp_path):
        """Batch classify should skip records that are already in wiki/."""
        raw_dir = tmp_path / "raw"
        wiki_dir = tmp_path / "wiki"
        raw_dir.mkdir()

        record = {
            "id": "raw_test_skip_001",
            "timestamp": "2026-08-05T01:00:00+05:30",
            "type": "note",
            "source": "test",
            "title": "Test Note",
            "raw_content": "Some reference content for testing skip behavior.",
            "tags": ["uncategorized"],
        }
        with open(raw_dir / "raw_test_skip_001.json", "w", encoding="utf-8") as f:
            json.dump(record, f)

        # First run
        created1 = batch_classify(raw_dir=raw_dir, wiki_dir=wiki_dir, provider="rule")
        assert len(created1) == 1

        # Second run (should skip)
        created2 = batch_classify(raw_dir=raw_dir, wiki_dir=wiki_dir, provider="rule")
        assert len(created2) == 0

    def test_force_reclassify(self, tmp_path):
        """With --force, batch classify should re-process already classified items."""
        raw_dir = tmp_path / "raw"
        wiki_dir = tmp_path / "wiki"
        raw_dir.mkdir()

        record = {
            "id": "raw_test_force_001",
            "timestamp": "2026-08-05T01:00:00+05:30",
            "type": "note",
            "source": "test",
            "title": "Force Test Note",
            "raw_content": "Testing forced reclassification of an already classified item.",
            "tags": ["uncategorized"],
        }
        with open(raw_dir / "raw_test_force_001.json", "w", encoding="utf-8") as f:
            json.dump(record, f)

        batch_classify(raw_dir=raw_dir, wiki_dir=wiki_dir, provider="rule")
        created2 = batch_classify(raw_dir=raw_dir, wiki_dir=wiki_dir, provider="rule", force=True)
        assert len(created2) == 1


# ---------------------------------------------------------------------------
# Classified IDs Tracker Tests
# ---------------------------------------------------------------------------

class TestGetClassifiedIds:
    def test_extracts_ids_from_wiki(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        resources_dir = wiki_dir / "3_Resources"
        resources_dir.mkdir(parents=True)

        md_content = '---\nid: "raw_20260805_abc12345"\ntitle: "Test"\n---\n# Test'
        (resources_dir / "test_note.md").write_text(md_content, encoding="utf-8")

        ids = _get_classified_ids(wiki_dir)
        assert "raw_20260805_abc12345" in ids

    def test_empty_wiki_returns_empty_set(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        for cat in PARA_CATEGORIES:
            (wiki_dir / cat).mkdir(parents=True)
        ids = _get_classified_ids(wiki_dir)
        assert len(ids) == 0
