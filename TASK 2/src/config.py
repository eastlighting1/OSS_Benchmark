from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset: str = "toy"
    documents_dir: Path = PROJECT_ROOT / "data" / "raw" / "documents"
    questions_csv: Path = PROJECT_ROOT / "data" / "raw" / "questions.csv"
    multihoprag_questions_json: Path = PROJECT_ROOT.parent / "data" / "MultiHopRAG.json"
    multihoprag_corpus_json: Path = PROJECT_ROOT.parent / "data" / "corpus.json"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    output_dir: Path = PROJECT_ROOT / "outputs"
    caracal_db_path: Path = Path("data") / "processed" / "graphrag_benchmark.crcl"
    vector_store_dir: Path = PROJECT_ROOT / "data" / "processed" / "vector_store"
    chunk_size_tokens: int = 80
    chunk_overlap_tokens: int = 12
    embedding_dimension: int = 64
    top_k_semantic_candidates: int = 8
    relation_depth: int = 2
    repeats: int = 3
    max_questions: int | None = None
    max_documents: int | None = None
    configs: tuple[str, ...] = field(
        default=("caracal-only", "neo4j-only", "caracal-external-semantic", "neo4j-external-semantic")
    )
