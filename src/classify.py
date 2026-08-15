import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.utils import slugify

load_dotenv()

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
RAW_DIR = BASE_DIR / "raw"
WIKI_DIR = BASE_DIR / "wiki"

PARA_CATEGORIES = ["1_Projects", "2_Areas", "3_Resources", "4_Archives"]

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "llama-3.1-8b-instant",
]

CLASSIFICATION_PROMPT = """You are a personal knowledge organizer. Classify the following captured item using the PARA methodology.

RULES:
1. Choose exactly ONE category:
   - "1_Projects": Time-bound active projects with specific goals, milestones, or deadlines.
   - "2_Areas": Long-term continuous responsibilities or standards (e.g., Health, Career, Finance, Software Architecture).
   - "3_Resources": Evergreen references, guides, research articles, book summaries, topic notes, or learning material.
   - "4_Archives": Inactive items, completed projects, deprecated docs, or obsolete material.

2. Priority hierarchy when ambiguous: Projects > Areas > Resources > Archives.

3. Extract 3-5 normalized lowercase English tags (even if the content is non-English).

4. Generate a concise one-line summary (max 120 characters).

5. Generate a clean, descriptive title (max 80 characters).

ITEM TYPE: {item_type}
ITEM SOURCE: {source}
ORIGINAL TITLE: {original_title}
CONTENT (first 2000 chars):
---
{content}
---

Respond with ONLY valid JSON (no markdown fences, no extra text):
{{"category": "1_Projects|2_Areas|3_Resources|4_Archives", "tags": ["tag1", "tag2", "tag3"], "summary": "One-line summary here", "title": "Clean Descriptive Title"}}"""


def _call_groq(prompt: str) -> str:
    from groq import Groq
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        raise ValueError("GROQ_API_KEY not configured")

    client = Groq(api_key=api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _invoke(model: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a JSON-only classifier. Respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    last_err = None
    for model in GROQ_MODELS:
        try:
            return _invoke(model)
        except Exception as e:
            last_err = e
            continue
    raise last_err  # type: ignore[misc]


def _call_gemini(prompt: str) -> str:
    from google import genai
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        raise ValueError("GEMINI_API_KEY not configured")

    client = genai.Client(api_key=api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _invoke() -> str:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return response.text

    return _invoke()


def _call_openai(prompt: str) -> str:
    from openai import OpenAI
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        raise ValueError("OPENAI_API_KEY not configured")

    client = OpenAI(api_key=api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _invoke() -> str:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a JSON-only classifier. Respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    return _invoke()


_PROJECT_KEYWORDS = [
    "deadline", "milestone", "sprint", "deliverable", "ship", "launch",
    "release", "roadmap", "mvp", "deploy", "build", "implement",
    "develop", "task", "goal", "plan", "project", "phase",
]
_AREA_KEYWORDS = [
    "health", "finance", "career", "habit", "routine", "responsibility",
    "standard", "maintenance", "ongoing", "practice", "workflow",
    "process", "system", "architecture", "infrastructure",
]
_ARCHIVE_KEYWORDS = [
    "deprecated", "obsolete", "archive", "old", "legacy", "inactive",
    "completed", "done", "finished", "retired",
]


def _rule_based_classify(record: Dict[str, Any]) -> Dict[str, Any]:
    content = (record.get("raw_content", "") + " " + record.get("title", "")).lower()
    words = set(re.findall(r"[a-z]+", content))

    project_score = len(words & set(_PROJECT_KEYWORDS))
    area_score = len(words & set(_AREA_KEYWORDS))
    archive_score = len(words & set(_ARCHIVE_KEYWORDS))

    if project_score >= 2:
        category = "1_Projects"
    elif area_score >= 2:
        category = "2_Areas"
    elif archive_score >= 2:
        category = "4_Archives"
    else:
        category = "3_Resources"

    word_freq: Dict[str, int] = {}
    for w in re.findall(r"[a-z]{4,}", content):
        if w not in {"this", "that", "with", "from", "your", "have", "been", "will",
                      "they", "them", "than", "into", "also", "more", "about", "when",
                      "what", "which", "their", "there", "other", "some", "each", "were",
                      "does", "make", "just", "over", "such", "like", "only", "most",
                      "very", "after", "before", "should", "could", "would", "being"}:
            word_freq[w] = word_freq.get(w, 0) + 1
    tags = sorted(word_freq, key=word_freq.get, reverse=True)[:5]  # type: ignore[arg-type]
    if not tags:
        tags = ["general"]

    title = record.get("title", "") or "Untitled"
    summary = (record.get("raw_content", "") or "")[:120].replace("\n", " ").strip()
    if not summary:
        summary = title

    return {
        "category": category,
        "tags": tags,
        "summary": summary,
        "title": title,
    }


def _extract_json(text: str) -> Dict[str, Any]:
    # This JSON parser was written in a delirium.
    # If the LLM returns markdown, malformed brackets, or alien signals,
    # this regex somehow finds the JSON. Only God knows how.
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from LLM response:\n{text[:500]}")


def classify_raw_item(
    raw_record: Dict[str, Any],
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    content_preview = (raw_record.get("raw_content", "") or "")[:2000]
    prompt = CLASSIFICATION_PROMPT.format(
        item_type=raw_record.get("type", "note"),
        source=raw_record.get("source", ""),
        original_title=raw_record.get("title", ""),
        content=content_preview,
    )

    providers: List[Tuple[str, Any]] = []
    if provider:
        provider_map = {
            "groq": ("groq", _call_groq),
            "gemini": ("gemini", _call_gemini),
            "openai": ("openai", _call_openai),
            "rule": ("rule", None),
        }
        if provider in provider_map:
            providers = [provider_map[provider]]
        else:
            raise ValueError(f"Unknown provider '{provider}'. Choose: groq, gemini, openai, rule")
    else:
        providers = [
            ("groq", _call_groq),
            ("gemini", _call_gemini),
            ("openai", _call_openai),
            ("rule", None),
        ]

    result: Optional[Dict[str, Any]] = None
    used_provider = "rule"

    for name, call_fn in providers:
        if name == "rule":
            result = _rule_based_classify(raw_record)
            used_provider = "rule"
            break
        try:
            raw_response = call_fn(prompt)
            result = _extract_json(raw_response)
            used_provider = name
            break
        except Exception as e:
            console.print(f"  [dim yellow]Provider {name} failed: {e}[/dim yellow]")
            continue

    if result is None:
        result = _rule_based_classify(raw_record)
        used_provider = "rule"

    category = result.get("category", "3_Resources")
    if category not in PARA_CATEGORIES:
        category = "3_Resources"

    tags = result.get("tags", ["general"])
    if not isinstance(tags, list) or len(tags) == 0:
        tags = ["general"]
    tags = [str(t).lower().strip().replace(" ", "-") for t in tags[:5]]

    summary = str(result.get("summary", ""))[:120].strip()
    if not summary:
        summary = (raw_record.get("raw_content", "") or "")[:120].replace("\n", " ").strip() or "No summary available"

    title = str(result.get("title", ""))[:80].strip()
    if not title:
        title = raw_record.get("title", "") or "Untitled"

    return {
        "category": category,
        "tags": tags,
        "summary": summary,
        "title": title,
        "provider": used_provider,
    }


def find_note_by_id(raw_id: str, wiki_dir: Optional[Path] = None) -> Optional[Path]:
    if not raw_id:
        return None
    target_wiki = wiki_dir or WIKI_DIR
    for category in PARA_CATEGORIES:
        cat_dir = target_wiki / category
        if not cat_dir.exists():
            continue
        for md_file in cat_dir.glob("*.md"):
            try:
                text = md_file.read_text(encoding="utf-8")
                match = re.search(r'^id:\s*"([^"]+)"', text, re.MULTILINE)
                if match and match.group(1) == raw_id:
                    return md_file
            except Exception:
                continue
    return None


def write_wiki_note(
    raw_record: Dict[str, Any],
    classification: Dict[str, Any],
    wiki_dir: Optional[Path] = None,
) -> Path:
    target_wiki = wiki_dir or WIKI_DIR
    category = classification["category"]
    category_dir = target_wiki / category
    category_dir.mkdir(parents=True, exist_ok=True)
    raw_id = raw_record.get("id", "")

    existing_note_path = find_note_by_id(raw_id, target_wiki) if raw_id else None
    title = classification["title"]

    if existing_note_path:
        if existing_note_path.parent.name == category:
            slug_path = existing_note_path
        else:
            slug_base = slugify(title)
            slug_path = category_dir / f"{slug_base}.md"
            counter = 1
            while slug_path.exists() and slug_path != existing_note_path:
                slug_path = category_dir / f"{slug_base}_{counter}.md"
                counter += 1
            try:
                existing_note_path.unlink()
            except Exception:
                pass
    else:
        slug_base = slugify(title)
        slug_path = category_dir / f"{slug_base}.md"
        counter = 1
        original_title = title
        while slug_path.exists():
            counter += 1
            title = f"{original_title} ({counter})"
            slug_base_new = slugify(title)
            slug_path = category_dir / f"{slug_base_new}.md"
        classification["title"] = title

    tags_yaml = "\n".join(f"  - {tag}" for tag in classification["tags"])
    created_at = raw_record.get("timestamp", datetime.now().astimezone().isoformat())

    frontmatter_lines = [
        "---",
        f'id: "{raw_record.get("id", "")}"',
        f'title: "{_escape_yaml_string(classification["title"])}"',
        f'category: "{category}"',
        f"tags:\n{tags_yaml}",
        f'summary: "{_escape_yaml_string(classification["summary"])}"',
        f'created_at: "{created_at}"',
        f'source: "{_escape_yaml_string(str(raw_record.get("source", "")))}"',
        f'type: "{raw_record.get("type", "note")}"',
        f'classified_by: "{classification.get("provider", "unknown")}"',
    ]

    attachment = raw_record.get("attachment")
    if attachment and isinstance(attachment, dict):
        frontmatter_lines.append(f'attachment_file: "{_escape_yaml_string(attachment.get("original_filename", ""))}"')

    frontmatter_lines.append("---")
    frontmatter = "\n".join(frontmatter_lines)

    raw_content = raw_record.get("raw_content", "").strip()
    source_info = raw_record.get("source", "")

    body_parts = [
        f"# {classification['title']}",
        "",
        f"> **Summary**: {classification['summary']}",
        "",
    ]

    if raw_record.get("type") == "link" and source_info:
        body_parts.append(f"**Source URL**: [{source_info}]({source_info})")
        body_parts.append("")

    if raw_record.get("type") == "file" and attachment:
        body_parts.append(f"**Original File**: `{attachment.get('original_filename', '')}`")
        body_parts.append("")

    body_parts.append("## Content")
    body_parts.append("")
    body_parts.append(raw_content if raw_content else "*No content extracted.*")
    body_parts.append("")

    full_content = frontmatter + "\n\n" + "\n".join(body_parts)

    with open(slug_path, "w", encoding="utf-8") as f:
        f.write(full_content)

    return slug_path


def _escape_yaml_string(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").strip()


def _get_classified_ids(wiki_dir: Optional[Path] = None) -> set:
    target_wiki = wiki_dir or WIKI_DIR
    classified = set()
    for category in PARA_CATEGORIES:
        cat_dir = target_wiki / category
        if not cat_dir.exists():
            continue
        for md_file in cat_dir.glob("*.md"):
            try:
                text = md_file.read_text(encoding="utf-8")
                match = re.search(r'^id:\s*"([^"]+)"', text, re.MULTILINE)
                if match:
                    classified.add(match.group(1))
            except Exception:
                continue
    return classified


def batch_classify(
    raw_dir: Optional[Path] = None,
    wiki_dir: Optional[Path] = None,
    force: bool = False,
    provider: Optional[str] = None,
    target_id: Optional[str] = None,
) -> List[Path]:
    target_raw = raw_dir or RAW_DIR
    target_wiki = wiki_dir or WIKI_DIR

    for cat in PARA_CATEGORIES:
        (target_wiki / cat).mkdir(parents=True, exist_ok=True)

    raw_files = sorted(target_raw.glob("*.json"))
    if not raw_files:
        console.print("[bold yellow]No raw captures found in raw/ directory.[/bold yellow]")
        return []

    if target_id:
        raw_files = [f for f in raw_files if f.stem == target_id]
        if not raw_files:
            console.print(f"[bold red]Raw capture '{target_id}' not found.[/bold red]")
            return []

    already_classified = set() if force else _get_classified_ids(target_wiki)

    created_paths: List[Path] = []
    results_table = Table(title="Classification Results", show_lines=True)
    results_table.add_column("ID", style="cyan", max_width=35)
    results_table.add_column("Title", style="white", max_width=40)
    results_table.add_column("Category", style="bold green", max_width=14)
    results_table.add_column("Tags", style="yellow", max_width=30)
    results_table.add_column("Provider", style="dim", max_width=10)

    skipped = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Classifying captures...", total=len(raw_files))

        for raw_file in raw_files:
            progress.update(task, description=f"Classifying {raw_file.stem[:30]}...")

            try:
                with open(raw_file, "r", encoding="utf-8") as f:
                    record = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                console.print(f"  [red]✗ Skipping {raw_file.name}: {e}[/red]")
                progress.advance(task)
                continue

            record_id = record.get("id", raw_file.stem)

            if record_id in already_classified:
                skipped += 1
                progress.advance(task)
                continue

            try:
                classification = classify_raw_item(record, provider=provider)
            except Exception as e:
                console.print(f"  [red]✗ Classification failed for {record_id}: {e}[/red]")
                progress.advance(task)
                continue

            try:
                wiki_path = write_wiki_note(record, classification, wiki_dir=target_wiki)
                created_paths.append(wiki_path)
                results_table.add_row(
                    record_id[:35],
                    classification["title"][:40],
                    classification["category"],
                    ", ".join(classification["tags"]),
                    classification.get("provider", "?"),
                )
            except Exception as e:
                console.print(f"  [red]✗ Failed to write wiki note for {record_id}: {e}[/red]")

            progress.advance(task)

    console.print()
    if created_paths:
        console.print(results_table)
    console.print(
        f"\n[bold green]Done! Classified {len(created_paths)} items[/bold green]"
        f" | [dim]{skipped} skipped[/dim]"
        f" | [dim]{len(raw_files) - len(created_paths) - skipped} errors[/dim]"
    )

    return created_paths


@click.group()
def main():
    pass


@main.command("run")
@click.option("--all", "classify_all", is_flag=True, default=True, help="Classify all raw captures.")
@click.option("--id", "target_id", default=None, help="Classify a single raw capture by ID.")
@click.option("--force", is_flag=True, default=False, help="Re-classify even if already processed.")
@click.option(
    "--provider",
    type=click.Choice(["groq", "gemini", "openai", "rule"], case_sensitive=False),
    default=None,
    help="Force a specific LLM provider instead of auto-fallback.",
)
def run_cmd(classify_all: bool, target_id: Optional[str], force: bool, provider: Optional[str]):
    console.print("\n[bold magenta]Running PARA Classifier...[/bold magenta]\n")
    batch_classify(force=force, provider=provider, target_id=target_id)


@main.command("status")
def status_cmd():
    raw_files = list(RAW_DIR.glob("*.json"))
    classified_ids = _get_classified_ids()

    raw_ids = set()
    for f in raw_files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                raw_ids.add(data.get("id", f.stem))
        except Exception:
            pass

    pending = raw_ids - classified_ids
    console.print(f"\n[bold]Classification Status[/bold]")
    console.print(f"  Raw captures:  {len(raw_ids)}")
    console.print(f"  Classified:    {len(classified_ids)}")
    console.print(f"  Pending:       {len(pending)}")

    for cat in PARA_CATEGORIES:
        cat_dir = WIKI_DIR / cat
        count = len(list(cat_dir.glob("*.md"))) if cat_dir.exists() else 0
        console.print(f"    - {cat}: {count} notes")
    console.print()


def cli_entrypoint():
    _fix_windows_encoding()
    if len(sys.argv) == 1:
        sys.argv.insert(1, "run")
    elif len(sys.argv) > 1:
        first_arg = sys.argv[1]
        valid_commands = ["run", "status", "--help", "-h"]
        if first_arg not in valid_commands:
            sys.argv.insert(1, "run")
    main()


if __name__ == "__main__":
    cli_entrypoint()
