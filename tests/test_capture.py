"""
Unit and integration tests for SecondSelf Capture Pipeline (Week 1 - The Archivist).
"""

import os
import json
import tempfile
from pathlib import Path
import pytest

from src.capture import capture_item, auto_capture, RAW_DIR, ensure_directories
from src.utils import generate_unique_id, get_timestamp, calculate_sha256, is_url


def test_unique_id_format():
    uid = generate_unique_id("raw")
    assert uid.startswith("raw_")
    parts = uid.split("_")
    assert len(parts) == 4
    assert len(parts[1]) == 8  # YYYYMMDD
    assert len(parts[2]) == 6  # HHMMSS
    assert len(parts[3]) == 8  # Short UUID


def test_get_timestamp():
    ts = get_timestamp()
    assert "T" in ts
    assert ":" in ts


def test_is_url():
    assert is_url("https://example.com") is True
    assert is_url("http://github.com/aryan1chauhan") is True
    assert is_url("not a url") is False
    assert is_url("C:\\path\\to\\file.txt") is False


def test_capture_note(tmp_path):
    raw_dir = tmp_path / "raw"
    record = capture_item("note", "This is a test note for SecondSelf.", title="Test Note Title", raw_dir=raw_dir)
    
    assert record["type"] == "note"
    assert record["title"] == "Test Note Title"
    assert record["raw_content"] == "This is a test note for SecondSelf."
    assert record["id"].startswith("raw_")
    
    saved_file = raw_dir / f"{record['id']}.json"
    assert saved_file.exists()
    
    with open(saved_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["id"] == record["id"]
        assert data["raw_content"] == record["raw_content"]


def test_capture_file(tmp_path):
    raw_dir = tmp_path / "raw"
    sample_file = tmp_path / "sample_doc.txt"
    sample_file.write_text("Sample document text payload.", encoding="utf-8")
    
    record = capture_item("file", str(sample_file), title="Sample Document", raw_dir=raw_dir)
    
    assert record["type"] == "file"
    assert record["title"] == "Sample Document"
    assert record["raw_content"] == "Sample document text payload."
    assert "attachment" in record
    assert record["attachment"]["original_filename"] == "sample_doc.txt"
    assert Path(record["attachment"]["absolute_stored_path"]).exists()


def test_auto_capture_detection(tmp_path):
    raw_dir = tmp_path / "raw"
    
    # Text note detection
    rec1 = capture_item("note", "Just plain text note", raw_dir=raw_dir)
    assert rec1["type"] == "note"
    
    # File path detection
    sample_file = tmp_path / "test.md"
    sample_file.write_text("# Test Markdown", encoding="utf-8")
    rec2 = capture_item("file", str(sample_file), raw_dir=raw_dir)
    assert rec2["type"] == "file"
