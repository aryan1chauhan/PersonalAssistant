"""
SecondSelf Utilities: ID generation, timestamping, hashing, web scraping, and file parsing.
"""

import os
import re
import hashlib
import uuid
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import requests
from bs4 import BeautifulSoup
try:
    import trafilatura
except ImportError:
    trafilatura = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None



def get_timestamp() -> str:
    """Generate ISO 8601 timestamp string with local timezone offset."""
    return datetime.now().astimezone().isoformat()


def generate_unique_id(prefix: str = "raw") -> str:
    """
    Generate unique ID in the format: prefix_YYYYMMDD_HHMMSS_8charUUID.
    Example: raw_20260805_012244_a1b2c3d4
    """
    now = datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"{prefix}_{date_str}_{short_uuid}"


def calculate_sha256(file_path: str) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def slugify(text: str, max_length: int = 50) -> str:
    """Convert text into a filesystem-safe clean slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    return text[:max_length].strip("_") or "untitled"


def is_url(text: str) -> bool:
    """Check if string is a HTTP/HTTPS URL."""
    url_pattern = re.compile(
        r"^(?:http|ftp)s?://"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )
    return bool(url_pattern.match(text.strip()))


def scrape_url(url: str) -> Dict[str, Any]:
    """
    Scrape web URL content using trafilatura with BeautifulSoup fallback.
    Returns dictionary with title, clean markdown text, and metadata.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()
        html_content = response.text
    except Exception as e:
        return {
            "title": f"Captured Link: {url}",
            "content": f"URL: {url}\n\n[Warning: Unable to fetch page content: {str(e)}]",
            "status": "failed",
            "error": str(e),
        }

    # Primary extraction using trafilatura if available
    extracted_text = None
    if trafilatura:
        try:
            extracted_text = trafilatura.extract(
                html_content, include_links=True, include_images=False, output_format="txt"
            )
        except Exception:
            extracted_text = None

    # Fallback to BeautifulSoup if available or regex
    title = url
    if BeautifulSoup:
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
            if not extracted_text:
                paragraphs = [p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip()]
                extracted_text = "\n\n".join(paragraphs) if paragraphs else None
        except Exception:
            pass

    if not extracted_text:
        extracted_text = re.sub(r"<[^>]+>", "", html_content).strip() or "No readable text extracted."

    return {
        "title": title,
        "content": extracted_text.strip(),
        "status": "success",
        "url": url,
    }


def parse_and_copy_file(file_path: str, assets_dir: str) -> Dict[str, Any]:
    """
    Parses a local file (PDF, TXT, MD), copies it to raw/assets/, and extracts text.
    """
    abs_path = Path(file_path).resolve()
    if not abs_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    filename = abs_path.name
    size_bytes = abs_path.stat().st_size
    sha256 = calculate_sha256(str(abs_path))
    
    # Ensure assets directory exists
    os.makedirs(assets_dir, exist_ok=True)
    
    # Target path in raw/assets/
    asset_id = generate_unique_id("asset")
    asset_filename = f"{asset_id}_{slugify(abs_path.stem)}{abs_path.suffix}"
    stored_path = os.path.join(assets_dir, asset_filename)
    shutil.copy2(str(abs_path), stored_path)

    extracted_text = ""
    ext = abs_path.suffix.lower()

    if ext == ".pdf":
        if pdfplumber:
            try:
                with pdfplumber.open(str(abs_path)) as pdf:
                    pages_text = [page.extract_text() or "" for page in pdf.pages]
                    extracted_text = "\n\n".join(pages_text).strip()
                if not extracted_text:
                    extracted_text = f"[PDF file captured: {filename}. Contains no selectable text / requires OCR]."
            except Exception as e:
                extracted_text = f"[PDF file captured: {filename}. Extraction error: {str(e)}]"
        else:
            extracted_text = f"[PDF file captured: {filename}. pdfplumber library not installed]."
    else:
        # Plain text / Markdown / Source files
        try:
            with open(str(abs_path), "r", encoding="utf-8", errors="ignore") as f:
                extracted_text = f.read().strip()
        except Exception as e:
            extracted_text = f"[File captured: {filename}. Error reading file: {str(e)}]"

    try:
        rel_stored_path = str(Path(stored_path).relative_to(Path.cwd()))
    except ValueError:
        rel_stored_path = str(stored_path)

    return {
        "title": f"File: {filename}",
        "content": extracted_text,
        "attachment": {
            "original_filename": filename,
            "stored_path": rel_stored_path,
            "absolute_stored_path": str(Path(stored_path).resolve()),
            "size_bytes": size_bytes,
            "sha256": sha256,
            "extension": ext,
        },
    }
