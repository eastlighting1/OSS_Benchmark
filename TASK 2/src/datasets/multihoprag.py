from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..models import BenchmarkQuestion, Document


def load_multihoprag_documents(
    corpus_path: Path,
    questions_path: Path | None = None,
    max_documents: int | None = None,
    max_questions: int | None = None,
) -> list[Document]:
    if not corpus_path.exists():
        raise FileNotFoundError(f"MultiHopRAG corpus does not exist: {corpus_path}")

    required_keys: set[str] = set()
    if questions_path is not None and questions_path.exists():
        for question in read_json_list(questions_path)[:max_questions]:
            for evidence in question.get("evidence_list") or []:
                required_keys.add(article_key(evidence))

    corpus = read_json_list(corpus_path)
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()

    for article in corpus:
        key = article_key(article)
        if key in required_keys:
            selected.append(article)
            selected_keys.add(key)

    for article in corpus:
        key = article_key(article)
        if key in selected_keys:
            continue
        if max_documents is not None and len(selected) >= max_documents:
            break
        selected.append(article)
        selected_keys.add(key)

    if not selected:
        raise ValueError(f"no MultiHopRAG documents loaded from {corpus_path}")

    return [article_to_document(article) for article in selected]


def load_multihoprag_questions(path: Path, max_questions: int | None = None) -> list[BenchmarkQuestion]:
    if not path.exists():
        raise FileNotFoundError(f"MultiHopRAG questions file does not exist: {path}")

    rows = read_json_list(path)
    questions: list[BenchmarkQuestion] = []
    for index, row in enumerate(rows[:max_questions]):
        evidence_list = row.get("evidence_list") or []
        questions.append(
            BenchmarkQuestion(
                question_id=f"q_mhr_{index + 1:04d}",
                question=normalize_text(str(row.get("query") or "")),
                answer=normalize_text(str(row.get("answer") or "")),
                question_type=normalize_text(str(row.get("question_type") or "")),
                gold_document_ids=unique_tuple(document_id_for_article(evidence) for evidence in evidence_list),
                gold_titles=unique_tuple(str(evidence.get("title") or "") for evidence in evidence_list),
                gold_urls=unique_tuple(str(evidence.get("url") or "") for evidence in evidence_list),
                gold_facts=unique_tuple(normalize_text(str(evidence.get("fact") or "")) for evidence in evidence_list),
            )
        )
    return [question for question in questions if question.question]


def article_to_document(article: dict[str, Any]) -> Document:
    title = normalize_text(str(article.get("title") or "Untitled MultiHopRAG article"))
    body = normalize_text(str(article.get("body") or ""))
    metadata_parts = [
        f"Title: {title}",
        f"Source: {normalize_text(str(article.get('source') or ''))}",
        f"Category: {normalize_text(str(article.get('category') or ''))}",
        f"Author: {normalize_text(str(article.get('author') or ''))}",
        f"Published: {normalize_text(str(article.get('published_at') or ''))}",
    ]
    text = normalize_text(" ".join([*metadata_parts, body]))
    return Document(
        document_id=document_id_for_article(article),
        title=title,
        source_path=str(article.get("url") or f"multihoprag:{article_key(article)}"),
        text=text,
        source_type="multihoprag",
    )


def read_json_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON list in {path}")
    return [item for item in data if isinstance(item, dict)]


def document_id_for_article(article: dict[str, Any]) -> str:
    digest = hashlib.sha1(article_key(article).encode("utf-8")).hexdigest()[:12]
    return f"doc:mhr:{digest}"


def article_key(article: dict[str, Any]) -> str:
    key = str(article.get("url") or article.get("title") or "").strip().lower()
    return " ".join(key.split())


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\r\n", "\n").split())


def unique_tuple(values) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = normalize_text(str(value))
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return tuple(result)
