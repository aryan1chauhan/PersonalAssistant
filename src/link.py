# link.py
# auto-links notes based on semantic similarity using sentence-transformers (all-MiniLM-L6-v2)
# computes cosine similarity matrix and appends bidirectional [[wikilinks]]

import os
import sys
import re
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from dotenv import load_dotenv

# fix windows encoding for terminal output
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

PARA_CATEGORIES = ["1_Projects", "2_Areas", "3_Resources", "4_Archives"]

# defaults (overridable via .env)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.65"))
MAX_LINKS_PER_NOTE = 5
MIN_WORD_COUNT = 15

RELATED_SECTION_HEADER = "## Related Knowledge"


def parse_wiki_note(filepath: Path) -> Optional[Dict[str, Any]]:
    # quick regex parser for YAML frontmatter so we don't need pyyaml
    try:
        text = filepath.read_text(encoding="utf-8")
    except Exception:
        return None

    fm_match = re.match(r"^---\s*\r?\n(.*?)\r?\n---", text, re.DOTALL)
    if not fm_match:
        return None

    frontmatter_text = fm_match.group(1)
    body_start = fm_match.end()
    body = text[body_start:].strip()

    # don't include previously generated related links in the embedding text
    body_for_embedding = re.split(
        r"^## Related Knowledge\s*$", body, flags=re.MULTILINE
    )[0].strip()

    def _extract_fm(key: str) -> str:
        match = re.search(rf'^{key}:\s*"?([^"\n]*)"?\s*$', frontmatter_text, re.MULTILINE)
        return match.group(1).strip() if match else ""

    title = _extract_fm("title")
    category = _extract_fm("category")
    summary = _extract_fm("summary")

    tags: List[str] = []
    tags_match = re.search(r"^tags:\s*\n((?:\s+-\s+.+\n?)+)", frontmatter_text, re.MULTILINE)
    if tags_match:
        tags = [t.strip() for t in re.findall(r"-\s+(.+)", tags_match.group(1))]

    words = re.findall(r"[a-zA-Z]+", body_for_embedding)
    word_count = len(words)

    return {
        "path": filepath,
        "title": title or filepath.stem.replace("_", " ").title(),
        "category": category,
        "tags": tags,
        "summary": summary,
        "body": body_for_embedding,
        "word_count": word_count,
    }


def scan_wiki_notes(wiki_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    target_wiki = wiki_dir or WIKI_DIR
    notes: List[Dict[str, Any]] = []

    for category in PARA_CATEGORIES:
        cat_dir = target_wiki / category
        if not cat_dir.exists():
            continue
        for md_file in sorted(cat_dir.glob("*.md")):
            if md_file.name == ".gitkeep":
                continue
            parsed = parse_wiki_note(md_file)
            if parsed:
                notes.append(parsed)

    return notes


def _load_embedding_model(model_name: Optional[str] = None):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for auto-linking. "
            "Install it with: pip install sentence-transformers"
        )

    model_id = model_name or EMBEDDING_MODEL
    return SentenceTransformer(model_id)


def _compose_embedding_text(note: Dict[str, Any]) -> str:
    # combine title + summary + body text for a fuller semantic representation
    parts = []
    if note.get("title"):
        parts.append(note["title"])
    if note.get("summary"):
        parts.append(note["summary"])
    if note.get("body"):
        parts.append(note["body"])
    return ". ".join(parts)


def compute_embeddings(
    notes: List[Dict[str, Any]],
    model=None,
    model_name: Optional[str] = None,
) -> List[Any]:
    if model is None:
        model = _load_embedding_model(model_name)

    texts = [_compose_embedding_text(note) for note in notes]
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return list(embeddings)


def cosine_similarity(vec_a, vec_b) -> float:
    import numpy as np

    a = np.asarray(vec_a, dtype=np.float64)
    b = np.asarray(vec_b, dtype=np.float64)

    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot / (norm_a * norm_b))


def compute_similarity_matrix(
    embeddings: List[Any],
) -> List[List[float]]:
    import numpy as np

    n = len(embeddings)
    matrix = [[0.0] * n for _ in range(n)]

    if n == 0:
        return matrix

    emb_array = np.array(embeddings, dtype=np.float64)
    norms = np.linalg.norm(emb_array, axis=1, keepdims=True)

    norms = np.where(norms == 0, 1.0, norms)
    normalized = emb_array / norms

    # dot product on normalized vectors = cosine similarity
    sim_matrix = np.dot(normalized, normalized.T)

    for i in range(n):
        for j in range(n):
            matrix[i][j] = float(sim_matrix[i][j])

    return matrix


def _extract_existing_wikilinks(text: str) -> set:
    return set(re.findall(r"\[\[([^\]]+)\]\]", text))


def _strip_related_section(text: str) -> str:
    parts = re.split(r"^## Related Knowledge\s*$", text, flags=re.MULTILINE)
    return parts[0].rstrip() + "\n"


def _build_related_section(linked_titles: List[str]) -> str:
    lines = [
        "",
        RELATED_SECTION_HEADER,
        "",
    ]
    for title in linked_titles:
        lines.append(f"- [[{title}]]")
    lines.append("")
    return "\n".join(lines)


def find_links(
    notes: List[Dict[str, Any]],
    similarity_matrix: List[List[float]],
    threshold: float = SIMILARITY_THRESHOLD,
    max_links: int = MAX_LINKS_PER_NOTE,
    min_words: int = MIN_WORD_COUNT,
) -> Dict[int, List[Tuple[int, float]]]:
    n = len(notes)
    links: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n)}

    # ignore super short stub notes to prevent noisy false positives
    short_indices = {i for i, note in enumerate(notes) if note["word_count"] < min_words}

    for i in range(n):
        if i in short_indices:
            continue

        candidates: List[Tuple[int, float]] = []
        for j in range(n):
            if i == j:
                continue
            if j in short_indices:
                continue

            sim = similarity_matrix[i][j]
            if sim >= threshold:
                candidates.append((j, sim))

        candidates.sort(key=lambda x: x[1], reverse=True)
        links[i] = candidates[:max_links]

    return links


def inject_wikilinks(
    notes: List[Dict[str, Any]],
    links: Dict[int, List[Tuple[int, float]]],
) -> int:
    # make links bidirectional so both notes reference each other
    bidir_links: Dict[int, set] = {i: set() for i in range(len(notes))}
    for i, targets in links.items():
        for j, _score in targets:
            bidir_links[i].add(j)
            bidir_links[j].add(i)

    modified_count = 0

    for i, note in enumerate(notes):
        target_indices = bidir_links[i]
        if not target_indices:
            continue

        filepath = note["path"]
        try:
            content = filepath.read_text(encoding="utf-8")
        except Exception:
            continue

        clean_content = _strip_related_section(content)

        existing_links = _extract_existing_wikilinks(clean_content)
        new_titles = []
        for j in sorted(target_indices):
            target_title = notes[j]["title"]
            if target_title not in existing_links:
                new_titles.append(target_title)

        if not new_titles:
            if RELATED_SECTION_HEADER in content:
                filepath.write_text(clean_content, encoding="utf-8")
                modified_count += 1
            continue

        related_section = _build_related_section(new_titles)
        new_content = clean_content + related_section

        filepath.write_text(new_content, encoding="utf-8")
        modified_count += 1

    return modified_count


def clear_all_related_sections(wiki_dir: Optional[Path] = None) -> int:
    target_wiki = wiki_dir or WIKI_DIR
    modified = 0

    for category in PARA_CATEGORIES:
        cat_dir = target_wiki / category
        if not cat_dir.exists():
            continue
        for md_file in cat_dir.glob("*.md"):
            if md_file.name == ".gitkeep":
                continue
            try:
                content = md_file.read_text(encoding="utf-8")
                if RELATED_SECTION_HEADER in content:
                    clean = _strip_related_section(content)
                    md_file.write_text(clean, encoding="utf-8")
                    modified += 1
            except Exception:
                continue

    return modified


def _save_embeddings(
    notes: List[Dict[str, Any]],
    embeddings: List[Any],
    data_dir: Optional[Path] = None,
) -> Path:
    import numpy as np

    target_dir = data_dir or DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = target_dir / "embeddings.pkl"

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
        "model": EMBEDDING_MODEL,
        "dimension": len(embeddings[0]) if embeddings else 0,
        "count": len(records),
        "notes": records,
    }

    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f)

    console.print(f"  Saved embeddings to [cyan]{pkl_path}[/cyan]")
    return pkl_path


def load_embeddings(data_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    target_dir = data_dir or DATA_DIR
    pkl_path = target_dir / "embeddings.pkl"

    if not pkl_path.exists():
        return None

    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def run_auto_linking(
    wiki_dir: Optional[Path] = None,
    model=None,
    model_name: Optional[str] = None,
    threshold: Optional[float] = None,
    max_links: int = MAX_LINKS_PER_NOTE,
    min_words: int = MIN_WORD_COUNT,
) -> Dict[str, Any]:
    target_wiki = wiki_dir or WIKI_DIR
    sim_threshold = threshold if threshold is not None else SIMILARITY_THRESHOLD

    # 1. scan notes
    console.print("[bold blue]Step 1/5:[/bold blue] Scanning wiki notes...")
    notes = scan_wiki_notes(target_wiki)
    if not notes:
        console.print("[bold yellow]No wiki notes found. Run classify.py first.[/bold yellow]")
        return {"total_notes": 0, "linked_notes": 0, "total_links": 0}

    console.print(f"  Found [cyan]{len(notes)}[/cyan] notes across PARA categories.")

    linkable_notes = [n for n in notes if n["word_count"] >= min_words]
    short_notes = [n for n in notes if n["word_count"] < min_words]
    if short_notes:
        console.print(
            f"  [dim]{len(short_notes)} notes skipped (< {min_words} words)[/dim]"
        )

    # 2. clear old links so we don't accumulate duplicates
    console.print("[bold blue]Step 2/5:[/bold blue] Clearing existing related sections...")
    cleared = clear_all_related_sections(target_wiki)
    if cleared:
        console.print(f"  Cleared [dim]{cleared}[/dim] existing sections.")
    notes = scan_wiki_notes(target_wiki)

    # 3. compute embeddings
    console.print(f"[bold blue]Step 3/5:[/bold blue] Computing embeddings with '{model_name or EMBEDDING_MODEL}'...")
    embeddings = compute_embeddings(notes, model=model, model_name=model_name)
    console.print(f"  Computed [cyan]{len(embeddings)}[/cyan] embedding vectors.")

    _save_embeddings(notes, embeddings, data_dir=DATA_DIR)

    # 4. compute similarity matrix
    console.print("[bold blue]Step 4/5:[/bold blue] Computing pairwise similarity...")
    sim_matrix = compute_similarity_matrix(embeddings)
    link_map = find_links(
        notes, sim_matrix,
        threshold=sim_threshold,
        max_links=max_links,
        min_words=min_words,
    )

    total_link_pairs = sum(len(targets) for targets in link_map.values())
    console.print(
        f"  Found [cyan]{total_link_pairs}[/cyan] directed link pairs "
        f"(threshold >= {sim_threshold})."
    )

    # 5. inject [[wikilinks]]
    console.print("[bold blue]Step 5/5:[/bold blue] Injecting bidirectional wikilinks...")
    modified = inject_wikilinks(notes, link_map)

    console.print()
    console.print(f"[bold green]Auto-linking complete![/bold green]")
    console.print(f"  Notes scanned:    {len(notes)}")
    console.print(f"  Notes linked:     {modified}")
    console.print(f"  Link pairs found: {total_link_pairs}")
    console.print(f"  Similarity threshold: {sim_threshold}")
    console.print()

    if total_link_pairs > 0:
        link_table = Table(title="Auto-Linked Note Pairs", show_lines=True)
        link_table.add_column("Source Note", style="cyan", max_width=35)
        link_table.add_column("-> Related Note", style="white", max_width=35)
        link_table.add_column("Similarity", style="green", max_width=10)

        for i, targets in link_map.items():
            for j, score in targets:
                link_table.add_row(
                    notes[i]["title"][:35],
                    notes[j]["title"][:35],
                    f"{score:.3f}",
                )
        console.print(link_table)

    return {
        "total_notes": len(notes),
        "linkable_notes": len(linkable_notes),
        "short_notes": len(short_notes),
        "linked_notes": modified,
        "total_links": total_link_pairs,
        "threshold": sim_threshold,
    }


@click.group()
def main():
    """Auto-link wiki notes using dense embeddings."""
    pass


@main.command("run")
@click.option(
    "--threshold", "-t", type=float, default=None,
    help=f"Cosine similarity threshold (default: {SIMILARITY_THRESHOLD}).",
)
@click.option(
    "--max-links", "-m", type=int, default=MAX_LINKS_PER_NOTE,
    help=f"Maximum links per note (default: {MAX_LINKS_PER_NOTE}).",
)
@click.option(
    "--min-words", "-w", type=int, default=MIN_WORD_COUNT,
    help=f"Minimum word count to include note (default: {MIN_WORD_COUNT}).",
)
def run_cmd(threshold: Optional[float], max_links: int, min_words: int):
    console.print("\n[bold magenta]Running Semantic Auto-Linker...[/bold magenta]\n")
    run_auto_linking(threshold=threshold, max_links=max_links, min_words=min_words)


@main.command("status")
def status_cmd():
    notes = scan_wiki_notes()
    if not notes:
        console.print("[bold yellow]No wiki notes found.[/bold yellow]")
        return

    linkable = [n for n in notes if n["word_count"] >= MIN_WORD_COUNT]
    short = [n for n in notes if n["word_count"] < MIN_WORD_COUNT]

    linked_count = 0
    total_links = 0
    for note in notes:
        try:
            content = note["path"].read_text(encoding="utf-8")
            if RELATED_SECTION_HEADER in content:
                linked_count += 1
                total_links += len(re.findall(r"\[\[([^\]]+)\]\]",
                    content.split(RELATED_SECTION_HEADER)[-1] if RELATED_SECTION_HEADER in content else ""))
        except Exception:
            pass

    console.print(f"\n[bold]Auto-Linking Status[/bold]")
    console.print(f"  Total wiki notes:     {len(notes)}")
    console.print(f"  Linkable (≥ {MIN_WORD_COUNT} words): {len(linkable)}")
    console.print(f"  Too short to link:    {len(short)}")
    console.print(f"  Notes with links:     {linked_count}")
    console.print(f"  Total wikilinks:      {total_links}")
    console.print(f"  Similarity threshold: {SIMILARITY_THRESHOLD}")
    console.print()

    for cat in PARA_CATEGORIES:
        cat_notes = [n for n in notes if n["category"] == cat]
        console.print(f"    {cat}: {len(cat_notes)} notes")
    console.print()


@main.command("clear")
@click.confirmation_option(prompt="Remove all ## Related Knowledge sections from wiki notes?")
def clear_cmd():
    console.print("\n[bold yellow]Clearing all Related Knowledge sections...[/bold yellow]")
    count = clear_all_related_sections()
    console.print(f"[bold green]Cleared {count} notes.[/bold green]\n")


@main.command("show")
@click.option("--full", is_flag=True, default=False, help="Show full embedding vectors.")
def show_cmd(full: bool):
    data = load_embeddings()
    if data is None:
        console.print("[bold yellow]No embeddings found. Run 'python link.py run' first.[/bold yellow]")
        return

    console.print(f"\n[bold]Stored Embeddings — data/embeddings.pkl[/bold]")
    console.print(f"  Model:     [cyan]{data['model']}[/cyan]")
    console.print(f"  Dimension: [cyan]{data['dimension']}[/cyan]")
    console.print(f"  Notes:     [cyan]{data['count']}[/cyan]\n")

    emb_table = Table(title="Embedding Records", show_lines=True)
    emb_table.add_column("#", style="dim", max_width=4)
    emb_table.add_column("Title", style="cyan", max_width=35)
    emb_table.add_column("Category", style="green", max_width=14)
    emb_table.add_column("Words", style="yellow", max_width=6)
    emb_table.add_column("Embedding (first 8 dims)", style="white", max_width=50)

    for idx, record in enumerate(data["notes"]):
        emb = record["embedding"]
        if full:
            emb_str = str(emb.tolist())
        else:
            preview = ", ".join(f"{v:.4f}" for v in emb[:8])
            emb_str = f"[{preview}, ...]"

        emb_table.add_row(
            str(idx),
            record["title"][:35],
            record["category"],
            str(record["word_count"]),
            emb_str,
        )

    console.print(emb_table)
    console.print()


def cli_entrypoint():
    _fix_windows_encoding()
    if len(sys.argv) == 1:
        sys.argv.insert(1, "run")
    elif len(sys.argv) > 1:
        first_arg = sys.argv[1]
        valid_commands = ["run", "status", "clear", "show", "--help", "-h"]
        if first_arg not in valid_commands and not first_arg.startswith("-"):
            sys.argv.insert(1, "run")
    main()


if __name__ == "__main__":
    cli_entrypoint()
