from __future__ import annotations

import csv
from pathlib import Path

from .datasets.multihoprag import load_multihoprag_questions
from .models import BenchmarkQuestion, Document


def load_documents(documents_dir: Path) -> list[Document]:
    if not documents_dir.exists():
        raise FileNotFoundError(f"documents folder does not exist: {documents_dir}")

    documents: list[Document] = []
    for path in sorted(documents_dir.glob("*")):
        if path.suffix.lower() not in {".md", ".txt"} or not path.is_file():
            continue
        text = normalize_text(path.read_text(encoding="utf-8"))
        if not text:
            continue
        documents.append(
            Document(
                document_id=stable_document_id(path),
                title=path.stem.replace("_", " ").title(),
                source_path=str(path),
                text=text,
                source_type=path.suffix.lower().lstrip("."),
            )
        )
    if not documents:
        raise ValueError(f"no Markdown or TXT documents found in {documents_dir}")
    return documents


def load_questions(path: Path) -> list[BenchmarkQuestion]:
    if not path.exists():
        return [
            BenchmarkQuestion(
                question_id="q_001",
                question="Why is GraphRAG useful for grounded question answering?",
            )
        ]
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        BenchmarkQuestion(
            question_id=row["question_id"],
            question=row["question"],
        )
        for row in rows
        if row.get("question")
    ]


def load_benchmark_questions(config) -> list[BenchmarkQuestion]:
    if config.dataset == "multihoprag":
        return load_multihoprag_questions(config.multihoprag_questions_json, config.max_questions)
    return load_questions(config.questions_csv)


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\r\n", "\n").split())


def stable_document_id(path: Path) -> str:
    return f"doc:{path.stem.lower().replace(' ', '_')}"
