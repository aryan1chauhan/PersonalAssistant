"""
SecondSelf Ingestion Module: Capture anything (note, link, file) into raw/ landing vault.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

import click
from rich.console import Console

from src.utils import (
    get_timestamp,
    generate_unique_id,
    is_url,
    scrape_url,
    parse_and_copy_file,
    slugify,
)

console = Console()

# Resolve workspace base paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw"
ASSETS_DIR = RAW_DIR / "assets"
WIKI_DIR = BASE_DIR / "wiki"


def ensure_directories():
    """Ensure raw/, raw/assets/, and wiki/ directories exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    for category in ["1_Projects", "2_Areas", "3_Resources", "4_Archives"]:
        (WIKI_DIR / category).mkdir(parents=True, exist_ok=True)


def capture_item(
    item_type: str,
    payload: str,
    title: Optional[str] = None,
    raw_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Core capture routine for notes, web links, and local files.
    Saves JSON record to raw/{id}.json and returns the record dict.
    """
    ensure_directories()
    target_raw_dir = raw_dir or RAW_DIR
    target_assets_dir = target_raw_dir / "assets"
    
    target_raw_dir.mkdir(parents=True, exist_ok=True)
    target_assets_dir.mkdir(parents=True, exist_ok=True)
    
    unique_id = generate_unique_id("raw")
    timestamp = get_timestamp()
    
    record = {
        "id": unique_id,
        "timestamp": timestamp,
        "type": item_type.lower(),
        "source": payload,
        "title": title or "",
        "raw_content": "",
        "tags": ["uncategorized", "raw_ingest"],
    }

    if item_type.lower() == "note":
        record["title"] = title or (payload[:40] + "..." if len(payload) > 40 else payload)
        record["raw_content"] = payload.strip()

    elif item_type.lower() == "link":
        console.print(f"[bold blue]Scraping URL:[/bold blue] {payload}")
        scraped = scrape_url(payload)
        record["title"] = title or scraped["title"]
        record["raw_content"] = scraped["content"]
        record["scrape_status"] = scraped.get("status", "success")

    elif item_type.lower() == "file":
        console.print(f"[bold blue]Parsing File:[/bold blue] {payload}")
        try:
            parsed = parse_and_copy_file(payload, str(target_assets_dir))
            record["title"] = title or parsed["title"]
            record["raw_content"] = parsed["content"]
            record["attachment"] = parsed["attachment"]
        except FileNotFoundError as e:
            console.print(f"[bold red]Error:[/bold red] File not found: '{payload}'. Please provide a valid file path.")
            sys.exit(1)

    else:
        raise ValueError(f"Unsupported item type: '{item_type}'. Must be note, link, or file.")

    # Save to raw/{id}.json
    out_file = target_raw_dir / f"{unique_id}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    console.print(
        f"[bold green][SUCCESS] Captured [{item_type.upper()}] successfully![/bold green]\n"
        f"  - ID: [cyan]{unique_id}[/cyan]\n"
        f"  - Title: {record['title']}\n"
        f"  - Saved: {out_file}"
    )

    return record


def auto_capture(payload: str, title: Optional[str] = None) -> Dict[str, Any]:
    """Auto-detect input modality (URL, local file, or text note)."""
    payload_str = payload.strip()

    if is_url(payload_str):
        return capture_item("link", payload_str, title=title)
    elif os.path.exists(payload_str):
        return capture_item("file", payload_str, title=title)
    else:
        return capture_item("note", payload_str, title=title)


# Click CLI Group
@click.group()
def main():
    """SecondSelf Capture Pipeline: Save any note, web link, or local file into raw/."""
    ensure_directories()


@main.command("note")
@click.argument("content")
@click.option("--title", "-t", help="Optional title for the note.")
def capture_note_cmd(content: str, title: Optional[str]):
    """Capture a plain text note."""
    capture_item("note", content, title=title)


@main.command("link")
@click.argument("url")
@click.option("--title", "-t", help="Optional title for the web link.")
def capture_link_cmd(url: str, title: Optional[str]):
    """Capture a web URL link."""
    capture_item("link", url, title=title)


@main.command("file")
@click.argument("filepath")
@click.option("--title", "-t", help="Optional title for the file.")
def capture_file_cmd(filepath: str, title: Optional[str]):
    """Capture a local document or PDF file."""
    capture_item("file", filepath, title=title)


@main.command("auto")
@click.argument("payload")
@click.option("--title", "-t", help="Optional title for captured item.")
def capture_auto_cmd(payload: str, title: Optional[str]):
    """Auto-detect modality (note, URL, or file path)."""
    auto_capture(payload, title=title)


def cli_entrypoint():
    """Smart CLI entrypoint allowing both subcommands and direct auto-detection."""
    ensure_directories()
    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        valid_commands = ["note", "link", "file", "auto", "--help", "-h"]
        if first_arg not in valid_commands and not first_arg.startswith("-"):
            # Insert 'auto' subcommand dynamically
            sys.argv.insert(1, "auto")
    main()


if __name__ == "__main__":
    cli_entrypoint()
