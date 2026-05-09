from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


Row = dict[str, Any]


@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    source_path: str
    text: str
    source_type: str = "markdown"


@dataclass(frozen=True)
class BenchmarkQuestion:
    question_id: str
    question: str
    answer: str = ""
    question_type: str = ""
    gold_document_ids: tuple[str, ...] = ()
    gold_titles: tuple[str, ...] = ()
    gold_urls: tuple[str, ...] = ()
    gold_facts: tuple[str, ...] = ()


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    token_count: int
    pagerank: float = 0.0
    community: int = -1


@dataclass(frozen=True)
class Entity:
    entity_id: str
    name: str
    canonical_name: str
    entity_type: str
    description: str = ""
    community: int = -1


@dataclass(frozen=True)
class EntityMention:
    chunk_id: str
    entity_id: str
    mention_text: str
    confidence: float = 1.0


@dataclass(frozen=True)
class Relationship:
    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    evidence_chunk_id: str
    weight: float = 1.0
    description: str = ""


@dataclass(frozen=True)
class EmbeddingRecord:
    owner_id: str
    owner_type: str
    vector: list[float]
    model_name: str = "deterministic-hash-v1"


@dataclass(frozen=True)
class QueryEntityLink:
    entity_id: str
    name: str
    score: float
    rank: int
    matched_text: str
    source: str


@dataclass(frozen=True)
class RetrievalPlan:
    question_type: str
    strategy: str
    semantic_top_k: int
    entity_top_k: int
    relation_depth: int
    evidence_budget: int
    citation_budget: int
    require_source_diversity: bool
    answer_mode: str


@dataclass
class GraphArtifacts:
    documents: list[Document]
    chunks: list[Chunk]
    entities: list[Entity]
    mentions: list[EntityMention]
    relationships: list[Relationship]
    embeddings: list[EmbeddingRecord]

    def counts(self) -> dict[str, int]:
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "entities": len(self.entities),
            "mentions": len(self.mentions),
            "relationships": len(self.relationships),
            "embeddings": len(self.embeddings),
        }


@dataclass(frozen=True)
class SemanticCandidate:
    node_id: str
    node_type: str
    score: float
    rank: int
    reason: str
    source: str


@dataclass(frozen=True)
class ContextItem:
    node_id: str
    node_type: str
    score: float
    reason: str
    path: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvidencePath:
    source_node_id: str
    target_node_id: str
    path: list[str]
    score: float


@dataclass(frozen=True)
class AnswerCandidate:
    candidate_text: str
    candidate_type: str
    score: float
    rank: int
    source_chunk_ids: tuple[str, ...]
    source_document_ids: tuple[str, ...]
    frequency: int
    reason: str


@dataclass(frozen=True)
class Answer:
    answer_id: str
    question_id: str
    question: str
    answer_text: str
    grounding_score: float
    config_id: str


@dataclass(frozen=True)
class Citation:
    citation_id: str
    answer_id: str
    chunk_id: str
    evidence_text: str
    confidence: float


@dataclass
class RetrievalTrace:
    run_id: str
    question_id: str
    question: str
    config_id: str
    semantic_entry_mode: str
    semantic_reentry_mode: str
    relation_expand_mode: str
    semantic_candidates: list[SemanticCandidate]
    context_items: list[ContextItem]
    evidence_paths: list[EvidencePath]
    query_entity_links: list[QueryEntityLink] = field(default_factory=list)
    retrieval_plan: RetrievalPlan | None = None
    predicted_answer: str = ""
    answer_candidates: list[AnswerCandidate] = field(default_factory=list)

    def to_json_dict(self) -> Row:
        return {
            "run_id": self.run_id,
            "question_id": self.question_id,
            "question": self.question,
            "config_id": self.config_id,
            "semantic_entry_mode": self.semantic_entry_mode,
            "semantic_reentry_mode": self.semantic_reentry_mode,
            "relation_expand_mode": self.relation_expand_mode,
            "semantic_candidates": [candidate.__dict__ for candidate in self.semantic_candidates],
            "context_items": [item.__dict__ for item in self.context_items],
            "evidence_paths": [path.__dict__ for path in self.evidence_paths],
            "query_entity_links": [link.__dict__ for link in self.query_entity_links],
            "retrieval_plan": None if self.retrieval_plan is None else self.retrieval_plan.__dict__,
            "predicted_answer": self.predicted_answer,
            "answer_candidates": [candidate.__dict__ for candidate in self.answer_candidates],
        }


@dataclass
class BenchmarkResult:
    config_id: str
    config_name: str
    graph_db: str
    vector_db: str
    status: str
    total_seconds: float | None
    load_seconds: float | None
    semantic_entry_seconds: float | None
    semantic_reentry_seconds: float | None
    relation_expansion_seconds: float | None
    trace_write_seconds: float | None
    query_planning_seconds: float | None
    query_entity_linking_seconds: float | None
    answer_aware_reranking_seconds: float | None
    documents: int | None
    chunks: int | None
    entities: int | None
    relationships: int | None
    retrieved_context_items: int | None
    citations: int | None
    adapter_loc: int
    semantic_entry_mode: str
    semantic_reentry_mode: str
    relation_expand_mode: str
    cross_store_join_count: int
    sync_notes: str
    notes: str


@dataclass(frozen=True)
class EvaluationResult:
    config_id: str
    question_id: str
    retrieval_precision_at_k: float
    context_recall: float
    citation_coverage: float
    answer_grounding_score: float
    context_items: int
    citations: int
    latency_seconds: float
    gold_evidence_documents: int = 0
    retrieved_gold_documents: int = 0
    evidence_recall_at_context: float = 0.0
    citation_recall: float = 0.0
    answer_exact_match: float = 0.0
    answer_contains_gold: float = 0.0
    answer_token_f1: float = 0.0
