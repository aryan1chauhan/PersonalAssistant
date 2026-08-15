# ask.py
# RAG query engine:
# 1. encodes question using sentence-transformers
# 2. finds closest wiki note snippets via cosine similarity
# 3. formats context and prompts LLM (Groq / Gemini / OpenAI) to answer with citations

import os
import sys
import re
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.link import (
    scan_wiki_notes,
    load_embeddings,
    cosine_similarity,
    compute_embeddings,
    _load_embedding_model,
    _compose_embedding_text,
    PARA_CATEGORIES,
    RELATED_SECTION_HEADER,
)

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
WIKI_DIR = BASE_DIR / "wiki"
DATA_DIR = BASE_DIR / "data"

DEFAULT_TOP_K = 5
MIN_RELEVANCE_SCORE = 0.25
MAX_CONTEXT_CHARS = 10000

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "llama-3.1-8b-instant",
]

RAG_SYSTEM_PROMPT = """You are SecondSelf, a personal knowledge assistant. You answer questions using ONLY the user's own notes provided as context below.

RULES:
1. Answer the question using ONLY information from the provided context passages.
2. If the context does not contain enough information to answer, say so clearly. Do NOT make up information.
3. Cite your sources by referencing the note titles in brackets, e.g. [Note Title].
4. Be concise but thorough. Summarize relevant information from multiple notes when applicable.
5. If multiple notes contain relevant information, synthesize a coherent answer from all of them.
6. Format your answer in clean Markdown for readability."""

RAG_USER_PROMPT = """## Context — Your Personal Notes

{context}

---

## Question

{question}

---

Answer the question above using ONLY the context from the personal notes provided. Cite specific note titles in [brackets] when referencing information."""


def _load_or_compute_embeddings(
    wiki_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], List[Any]]:
    import numpy as np

    target_data = data_dir or DATA_DIR
    target_wiki = wiki_dir or WIKI_DIR

    cached = load_embeddings(target_data)

    if cached and cached.get("notes"):
        notes_meta = cached["notes"]
        embeddings = [record["embedding"] for record in notes_meta]
        return notes_meta, embeddings

    # no cache on disk yet, generate now
    console.print("[dim]No cached embeddings found. Computing fresh embeddings...[/dim]")
    notes = scan_wiki_notes(target_wiki)
    if not notes:
        return [], []

    model = _load_embedding_model()
    embeddings = compute_embeddings(notes, model=model)

    notes_meta = []
    for i, note in enumerate(notes):
        notes_meta.append({
            "title": note["title"],
            "category": note["category"],
            "tags": note.get("tags", []),
            "summary": note.get("summary", ""),
            "path": str(note["path"]),
            "word_count": note["word_count"],
            "embedding": np.array(embeddings[i]),
        })

    return notes_meta, embeddings


def retrieve_relevant_notes(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = MIN_RELEVANCE_SCORE,
    wiki_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    import numpy as np

    notes_meta, embeddings = _load_or_compute_embeddings(wiki_dir, data_dir)

    if not notes_meta:
        return []

    model = _load_embedding_model()
    question_embedding = model.encode([question], show_progress_bar=False, convert_to_numpy=True)[0]

    scored_notes: List[Tuple[float, Dict[str, Any]]] = []
    for i, note_meta in enumerate(notes_meta):
        note_emb = np.array(note_meta["embedding"])
        sim = cosine_similarity(question_embedding, note_emb)

        if sim >= min_score:
            scored_notes.append((sim, note_meta))

    scored_notes.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, meta in scored_notes[:top_k]:
        full_body = _read_note_body(meta["path"])

        # for long notes, extract top chunks instead of blindly cutting off
        if len(full_body) > 4000:
            content_snippet = _extract_best_chunks(
                full_body, question_embedding, model, max_chars=4000
            )
        else:
            content_snippet = full_body

        results.append({
            "title": meta["title"],
            "category": meta["category"],
            "tags": meta.get("tags", []),
            "summary": meta.get("summary", ""),
            "path": meta["path"],
            "similarity_score": score,
            "content_snippet": content_snippet,
        })

    return results


def _read_note_body(filepath_str: str) -> str:
    try:
        filepath = Path(filepath_str)
        if not filepath.exists():
            return ""

        text = filepath.read_text(encoding="utf-8")

        fm_match = re.match(r"^---\s*\r?\n(.*?)\r?\n---", text, re.DOTALL)
        if fm_match:
            body = text[fm_match.end():].strip()
        else:
            body = text.strip()

        body = re.split(
            r"^## Related Knowledge\s*$", body, flags=re.MULTILINE
        )[0].strip()

        return body

    except Exception:
        return ""


def _read_note_content(filepath_str: str, max_chars: int = 4000) -> str:
    body = _read_note_body(filepath_str)
    if len(body) > max_chars:
        body = body[:max_chars] + "\n\n[...content truncated...]"
    return body


def _chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            boundary = text.rfind("\n\n", start + chunk_size // 2, end)
            if boundary == -1:
                boundary = text.rfind(". ", start + chunk_size // 2, end)
                if boundary != -1:
                    boundary += 2
            if boundary != -1:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap if end < len(text) else len(text)

    return chunks


def _extract_best_chunks(
    body: str,
    question_embedding,
    model,
    max_chars: int = 4000,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> str:
    import numpy as np

    chunks = _chunk_text(body, chunk_size, chunk_overlap)
    if not chunks:
        return ""
    if len(chunks) == 1:
        return chunks[0][:max_chars]

    chunk_embeddings = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)

    scored: List[Tuple[float, int, str]] = []
    for idx, (chunk, chunk_emb) in enumerate(zip(chunks, chunk_embeddings)):
        sim = cosine_similarity(question_embedding, np.array(chunk_emb))
        scored.append((sim, idx, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected_indices: List[int] = []
    total_chars = 0
    for sim, idx, chunk in scored:
        if total_chars + len(chunk) > max_chars and selected_indices:
            break
        selected_indices.append(idx)
        total_chars += len(chunk)

    selected_indices.sort()

    parts = [chunks[idx] for idx in selected_indices]

    if len(parts) > 1:
        return "\n\n[...]\n\n".join(parts)
    return parts[0] if parts else ""


def build_context_block(
    retrieved_notes: List[Dict[str, Any]],
    max_total_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    if not retrieved_notes:
        return "(No relevant notes found in your knowledge base.)"

    context_parts = []
    char_count = 0

    for i, note in enumerate(retrieved_notes, 1):
        header = (
            f"### Source {i}: {note['title']}\n"
            f"**Category**: {note['category']} | "
            f"**Tags**: {', '.join(note.get('tags', []))} | "
            f"**Relevance**: {note['similarity_score']:.1%}\n"
            f"**File**: `{Path(note['path']).name}`\n\n"
        )

        content = note.get("content_snippet", note.get("summary", ""))
        block = header + content + "\n"

        if char_count + len(block) > max_total_chars and context_parts:
            break

        context_parts.append(block)
        char_count += len(block)

    return "\n---\n\n".join(context_parts)


def _synthesize_groq(system_prompt: str, user_prompt: str) -> str:
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
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1500,
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


def _synthesize_gemini(system_prompt: str, user_prompt: str) -> str:
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
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=full_prompt,
        )
        return response.text

    return _invoke()


def _synthesize_openai(system_prompt: str, user_prompt: str) -> str:
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
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1500,
        )
        return response.choices[0].message.content

    return _invoke()


def ask(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = MIN_RELEVANCE_SCORE,
    provider: Optional[str] = None,
    wiki_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    retrieved = retrieve_relevant_notes(
        question=question,
        top_k=top_k,
        min_score=min_score,
        wiki_dir=wiki_dir,
        data_dir=data_dir,
    )

    if verbose:
        console.print(f"\n[dim]Retrieved {len(retrieved)} relevant notes for: \"{question}\"[/dim]")
        for note in retrieved:
            console.print(
                f"  [dim]• {note['title']} "
                f"({note['category']}) — "
                f"similarity: {note['similarity_score']:.3f}[/dim]"
            )

    context_block = build_context_block(retrieved)
    user_prompt = RAG_USER_PROMPT.format(
        context=context_block,
        question=question,
    )

    providers: List[Tuple[str, Any]] = []
    if provider:
        provider_map = {
            "groq": ("groq", _synthesize_groq),
            "gemini": ("gemini", _synthesize_gemini),
            "openai": ("openai", _synthesize_openai),
        }
        if provider in provider_map:
            providers = [provider_map[provider]]
        else:
            raise ValueError(f"Unknown provider '{provider}'. Choose: groq, gemini, openai")
    else:
        # fallback: groq -> gemini -> openai
        providers = [
            ("groq", _synthesize_groq),
            ("gemini", _synthesize_gemini),
            ("openai", _synthesize_openai),
        ]

    answer = None
    used_provider = "none"

    for name, synthesize_fn in providers:
        try:
            answer = synthesize_fn(RAG_SYSTEM_PROMPT, user_prompt)
            used_provider = name
            break
        except Exception as e:
            if verbose:
                console.print(f"  [dim yellow]Provider {name} failed: {e}[/dim yellow]")
            continue

    if answer is None:
        if not retrieved:
            answer = (
                "I couldn't find any relevant notes in your knowledge base to answer "
                "this question. Try capturing more notes first, or rephrase your question."
            )
        else:
            answer = (
                "I found relevant notes but couldn't connect to any LLM provider to "
                "synthesize an answer. Please check your API keys in .env.\n\n"
                "**Relevant notes found:**\n" +
                "\n".join(f"- [{n['title']}] (similarity: {n['similarity_score']:.1%})" for n in retrieved)
            )
        used_provider = "none"

    sources = [
        {
            "title": note["title"],
            "path": note["path"],
            "category": note["category"],
            "tags": note.get("tags", []),
            "similarity_score": note["similarity_score"],
        }
        for note in retrieved
    ]

    return {
        "answer": answer,
        "sources": sources,
        "question": question,
        "provider": used_provider,
        "retrieval_count": len(retrieved),
    }


def _display_answer(result: Dict[str, Any]) -> None:
    console.print()

    answer_md = Markdown(result["answer"])
    console.print(Panel(
        answer_md,
        title="[bold green]Answer[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))

    if result["sources"]:
        source_table = Table(title="Sources", show_lines=True)
        source_table.add_column("#", style="dim", max_width=4)
        source_table.add_column("Note Title", style="cyan", max_width=40)
        source_table.add_column("Category", style="green", max_width=14)
        source_table.add_column("Relevance", style="yellow", max_width=10)
        source_table.add_column("File", style="dim", max_width=35)

        for i, src in enumerate(result["sources"], 1):
            source_table.add_row(
                str(i),
                src["title"][:40],
                src["category"],
                f"{src['similarity_score']:.1%}",
                Path(src["path"]).name,
            )

        console.print(source_table)

    console.print(
        f"\n[dim]Provider: {result['provider']} | "
        f"Notes retrieved: {result['retrieval_count']}[/dim]\n"
    )


@click.group()
def main():
    """Ask questions over your notes with RAG."""
    pass


@main.command("query")
@click.argument("question")
@click.option(
    "--top-k", "-k", type=int, default=DEFAULT_TOP_K,
    help=f"Number of top matching notes to retrieve (default: {DEFAULT_TOP_K}).",
)
@click.option(
    "--min-score", "-s", type=float, default=MIN_RELEVANCE_SCORE,
    help=f"Minimum similarity score for relevance (default: {MIN_RELEVANCE_SCORE}).",
)
@click.option(
    "--provider", "-p",
    type=click.Choice(["groq", "gemini", "openai"], case_sensitive=False),
    default=None,
    help="Force a specific LLM provider instead of auto-fallback.",
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Show retrieval details.")
def query_cmd(question: str, top_k: int, min_score: float, provider: Optional[str], verbose: bool):
    console.print(f"[bold]Question:[/bold] {question}\n")

    result = ask(
        question=question,
        top_k=top_k,
        min_score=min_score,
        provider=provider,
        verbose=verbose,
    )

    _display_answer(result)


@main.command("search")
@click.argument("question")
@click.option(
    "--top-k", "-k", type=int, default=DEFAULT_TOP_K,
    help=f"Number of top matching notes to retrieve (default: {DEFAULT_TOP_K}).",
)
@click.option(
    "--min-score", "-s", type=float, default=MIN_RELEVANCE_SCORE,
    help=f"Minimum similarity score for relevance (default: {MIN_RELEVANCE_SCORE}).",
)
def search_cmd(question: str, top_k: int, min_score: float):
    console.print(f"[bold]Query:[/bold] {question}\n")

    retrieved = retrieve_relevant_notes(
        question=question,
        top_k=top_k,
        min_score=min_score,
    )

    if not retrieved:
        console.print("[bold yellow]No relevant notes found for this query.[/bold yellow]\n")
        return

    result_table = Table(title=f"Search Results ({len(retrieved)} matches)", show_lines=True)
    result_table.add_column("#", style="dim", max_width=4)
    result_table.add_column("Note Title", style="cyan", max_width=35)
    result_table.add_column("Category", style="green", max_width=14)
    result_table.add_column("Tags", style="yellow", max_width=25)
    result_table.add_column("Similarity", style="bold white", max_width=10)
    result_table.add_column("Summary", style="dim", max_width=50)

    for i, note in enumerate(retrieved, 1):
        result_table.add_row(
            str(i),
            note["title"][:35],
            note["category"],
            ", ".join(note.get("tags", []))[:25],
            f"{note['similarity_score']:.3f}",
            note.get("summary", "")[:50],
        )

    console.print(result_table)
    console.print()


@main.command("status")
def status_cmd():
    console.print("\n[bold]RAG Engine Status[/bold]\n")

    cached = load_embeddings()
    if cached and cached.get("notes"):
        note_count = len(cached["notes"])
        model_name = cached.get("model", "unknown")
        dimension = cached.get("dimension", "?")
        console.print(f"  Embeddings:    [green]{note_count} notes indexed[/green]")
        console.print(f"  Model:         [cyan]{model_name}[/cyan]")
        console.print(f"  Dimension:     [cyan]{dimension}[/cyan]")
    else:
        console.print("  Embeddings:    [red]Not computed[/red]")
        console.print("  [dim]Run 'python link.py run' to compute embeddings first.[/dim]")

    notes = scan_wiki_notes()
    console.print(f"  Wiki notes:    [cyan]{len(notes)}[/cyan]")
    for cat in PARA_CATEGORIES:
        cat_notes = [n for n in notes if n["category"] == cat]
        if cat_notes:
            console.print(f"    {cat}: {len(cat_notes)} notes")

    console.print()
    console.print("  LLM Providers:")
    groq_key = os.getenv("GROQ_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    for name, key in [("Groq", groq_key), ("Gemini", gemini_key), ("OpenAI", openai_key)]:
        if key and not key.startswith("your_"):
            console.print(f"    {name}: [green]configured[/green]")
        else:
            console.print(f"    {name}: [dim]not configured[/dim]")

    console.print()


def cli_entrypoint():
    _fix_windows_encoding()
    if len(sys.argv) == 1:
        sys.argv.insert(1, "status")
    elif len(sys.argv) > 1:
        first_arg = sys.argv[1]
        valid_commands = ["query", "search", "status", "--help", "-h"]
        if first_arg not in valid_commands and not first_arg.startswith("-"):
            sys.argv.insert(1, "query")
    main()


if __name__ == "__main__":
    cli_entrypoint()
