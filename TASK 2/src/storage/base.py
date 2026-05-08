from __future__ import annotations

import abc
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

from ..embeddings import cosine_similarity
from ..models import (
    Answer,
    Citation,
    ContextItem,
    EmbeddingRecord,
    Entity,
    EvidencePath,
    GraphArtifacts,
    QueryEntityLink,
    RetrievalPlan,
    RetrievalTrace,
    SemanticCandidate,
)


class MissingExternalService(RuntimeError):
    """Raised when an optional benchmark service is not configured."""


@dataclass(frozen=True)
class EdgeRecord:
    src: str
    dst: str
    edge_type: str
    weight: float
    evidence_chunk_id: str | None = None


@dataclass(frozen=True)
class NativeGraphRetrieval:
    semantic_candidates: list[SemanticCandidate]
    query_entity_links: list[QueryEntityLink]
    context_items: list[ContextItem]
    operator_timings_ms: dict[str, float]
    profile: dict[str, object]


class StorageAdapter(abc.ABC):
    config_id = "base"
    config_name = "Base"
    graph_db = "none"
    vector_db = "none"
    semantic_entry_mode = "not_run"
    semantic_reentry_mode = "not_run"
    relation_expand_mode = "not_run"

    def __init__(self) -> None:
        self.artifacts: GraphArtifacts | None = None
        self.embeddings_by_owner: dict[str, EmbeddingRecord] = {}
        self.entities_by_id: dict[str, Entity] = {}
        self.edges: list[EdgeRecord] = []
        self.adjacency: dict[str, list[EdgeRecord]] = {}
        self.traces: list[RetrievalTrace] = []
        self.answers: list[Answer] = []
        self.citations: list[Citation] = []
        self.cross_store_join_count = 0
        self.sync_notes = ""
        self.notes = ""

    def load(self, artifacts: GraphArtifacts) -> None:
        self.artifacts = artifacts
        self.embeddings_by_owner = {record.owner_id: record for record in artifacts.embeddings}
        self.entities_by_id = {entity.entity_id: entity for entity in artifacts.entities}
        self.edges = build_edges(artifacts)
        self.adjacency = build_adjacency(self.edges)

    def semantic_entry(
        self,
        question: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[SemanticCandidate]:
        self.semantic_entry_mode = "exact_scan"
        return exact_semantic_candidates(
            records=self.embeddings_by_owner.values(),
            query_embedding=query_embedding,
            top_k=top_k,
            source=self.semantic_entry_mode,
            reason="embedding similarity exact scan",
        )

    def semantic_reentry(self, candidates: list[SemanticCandidate]) -> list[SemanticCandidate]:
        self.semantic_reentry_mode = "native_result_nodes"
        return candidates

    def native_graph_retrieval(
        self,
        question: str,
        query_embedding: list[float],
        plan: RetrievalPlan,
    ) -> NativeGraphRetrieval | None:
        return None

    def link_query_entities(
        self,
        question: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[QueryEntityLink]:
        if self.artifacts is None:
            return []
        return link_query_entities_from_entities(
            question=question,
            entities=self.artifacts.entities,
            embeddings_by_owner=self.embeddings_by_owner,
            query_embedding=query_embedding,
            top_k=top_k,
            source="memory_entity_linking",
        )

    def evidence_path_expand(
        self,
        semantic_candidates: list[SemanticCandidate],
        entity_links: list[QueryEntityLink],
        plan: RetrievalPlan,
    ) -> list[ContextItem]:
        seed_node_ids = [
            *[candidate.node_id for candidate in semantic_candidates],
            *[link.entity_id for link in entity_links],
        ]
        return self.relation_expand(unique_preserving_order(seed_node_ids), plan.relation_depth)

    def relation_expand(self, seed_node_ids: list[str], depth: int) -> list[ContextItem]:
        self.relation_expand_mode = "memory_bfs_fallback"
        return bfs_context(seed_node_ids, self.adjacency, depth)

    def build_evidence_paths(self, context_items: list[ContextItem]) -> list[EvidencePath]:
        paths: list[EvidencePath] = []
        for item in context_items:
            if len(item.path) >= 2:
                paths.append(
                    EvidencePath(
                        source_node_id=item.path[0],
                        target_node_id=item.node_id,
                        path=item.path,
                        score=item.score,
                    )
                )
        return paths

    def store_retrieval_trace(self, trace: RetrievalTrace) -> None:
        self.traces.append(trace)

    def store_answer(self, answer: Answer, citations: list[Citation]) -> None:
        self.answers.append(answer)
        self.citations.extend(citations)

    def close(self) -> None:
        return


def exact_semantic_candidates(
    records: Iterable[EmbeddingRecord],
    query_embedding: list[float],
    top_k: int,
    source: str,
    reason: str,
) -> list[SemanticCandidate]:
    scored: list[tuple[str, str, float]] = []
    for record in records:
        score = cosine_similarity(query_embedding, record.vector)
        scored.append((record.owner_id, record.owner_type, score))
    scored.sort(key=lambda item: (-item[2], item[0]))
    return [
        SemanticCandidate(
            node_id=owner_id,
            node_type=owner_type,
            score=float(score),
            rank=index + 1,
            reason=reason,
            source=source,
        )
        for index, (owner_id, owner_type, score) in enumerate(scored[:top_k])
    ]


def link_query_entities_from_entities(
    question: str,
    entities: Iterable[Entity],
    embeddings_by_owner: dict[str, EmbeddingRecord],
    query_embedding: list[float],
    top_k: int,
    source: str,
    use_vector_fallback: bool = True,
) -> list[QueryEntityLink]:
    question_lower = question.lower()
    question_tokens = significant_tokens(question)
    scored: dict[str, QueryEntityLink] = {}

    for entity in entities:
        name_lower = entity.name.lower()
        entity_tokens = significant_tokens(entity.name)
        exact = 1.0 if name_lower and name_lower in question_lower else 0.0
        overlap = len(question_tokens & entity_tokens) / max(1, len(entity_tokens))
        score = exact + 0.6 * overlap
        if score <= 0:
            continue
        scored[entity.entity_id] = QueryEntityLink(
            entity_id=entity.entity_id,
            name=entity.name,
            score=score,
            rank=0,
            matched_text=entity.name if exact else " ".join(sorted(question_tokens & entity_tokens)),
            source=source,
        )

    if use_vector_fallback and len(scored) < top_k:
        vector_links: list[tuple[Entity, float]] = []
        for entity in entities:
            record = embeddings_by_owner.get(entity.entity_id)
            if record is None:
                continue
            vector_links.append((entity, cosine_similarity(query_embedding, record.vector)))
        vector_links.sort(key=lambda item: (-item[1], item[0].entity_id))
        for entity, score in vector_links[: top_k * 2]:
            if entity.entity_id in scored:
                continue
            if score <= 0:
                continue
            scored[entity.entity_id] = QueryEntityLink(
                entity_id=entity.entity_id,
                name=entity.name,
                score=0.25 + score,
                rank=0,
                matched_text="embedding_neighborhood",
                source=f"{source}+entity_embedding",
            )
            if len(scored) >= top_k:
                break

    ordered = sorted(scored.values(), key=lambda link: (-link.score, link.entity_id))[:top_k]
    return [
        QueryEntityLink(
            entity_id=link.entity_id,
            name=link.name,
            score=link.score,
            rank=index + 1,
            matched_text=link.matched_text,
            source=link.source,
        )
        for index, link in enumerate(ordered)
    ]


def build_edges(artifacts: GraphArtifacts) -> list[EdgeRecord]:
    edges: list[EdgeRecord] = []
    edges.extend(
        EdgeRecord(document.document_id, chunk.chunk_id, "HAS_CHUNK", 1.0)
        for document in artifacts.documents
        for chunk in artifacts.chunks
        if chunk.document_id == document.document_id
    )
    edges.extend(
        EdgeRecord(mention.chunk_id, mention.entity_id, "MENTIONS", mention.confidence)
        for mention in artifacts.mentions
    )
    for relationship in artifacts.relationships:
        edges.append(
            EdgeRecord(
                relationship.source_entity_id,
                relationship.target_entity_id,
                relationship.relationship_type,
                relationship.weight,
                relationship.evidence_chunk_id,
            )
        )
        edges.append(
            EdgeRecord(
                relationship.source_entity_id,
                relationship.evidence_chunk_id,
                "EVIDENCED_BY",
                relationship.weight,
                relationship.evidence_chunk_id,
            )
        )
        edges.append(
            EdgeRecord(
                relationship.target_entity_id,
                relationship.evidence_chunk_id,
                "EVIDENCED_BY",
                relationship.weight,
                relationship.evidence_chunk_id,
            )
        )
    return edges


def build_adjacency(edges: list[EdgeRecord]) -> dict[str, list[EdgeRecord]]:
    adjacency: dict[str, list[EdgeRecord]] = defaultdict(list)
    for edge in edges:
        adjacency[edge.src].append(edge)
        adjacency[edge.dst].append(
            EdgeRecord(
                src=edge.dst,
                dst=edge.src,
                edge_type=edge.edge_type,
                weight=edge.weight,
                evidence_chunk_id=edge.evidence_chunk_id,
            )
        )
    return dict(adjacency)


def bfs_context(seed_node_ids: list[str], adjacency: dict[str, list[EdgeRecord]], depth: int) -> list[ContextItem]:
    seen_depth: dict[str, int] = {}
    queue: deque[tuple[str, int, list[str], float]] = deque()
    for seed in seed_node_ids:
        queue.append((seed, 0, [seed], 1.0))
        seen_depth[seed] = 0

    context_by_node: dict[str, ContextItem] = {}
    while queue:
        node_id, current_depth, path, path_weight = queue.popleft()
        node_type = node_id.split(":", 1)[0].title()
        if node_id.startswith("chunk:"):
            context_by_node[node_id] = ContextItem(
                node_id=node_id,
                node_type="Chunk",
                score=path_weight / max(1, current_depth + 1),
                reason=f"relation path depth={current_depth}",
                path=path,
            )
        if current_depth >= depth:
            continue
        for edge in adjacency.get(node_id, []):
            next_depth = current_depth + 1
            previous_depth = seen_depth.get(edge.dst)
            if previous_depth is not None and previous_depth <= next_depth:
                continue
            seen_depth[edge.dst] = next_depth
            queue.append((edge.dst, next_depth, [*path, edge.edge_type, edge.dst], path_weight * edge.weight))

    return sorted(context_by_node.values(), key=lambda item: (-item.score, item.node_id))


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "both",
    "by",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "reported",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "which",
    "who",
    "with",
}


def significant_tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text) if token.lower() not in STOPWORDS and len(token) > 2}


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def merge_query_entity_links(*groups: Iterable[QueryEntityLink], top_k: int) -> list[QueryEntityLink]:
    merged: dict[str, QueryEntityLink] = {}
    for group in groups:
        for link in group:
            previous = merged.get(link.entity_id)
            if previous is None or link.score > previous.score:
                merged[link.entity_id] = link
    ordered = sorted(merged.values(), key=lambda value: (-value.score, value.entity_id))[:top_k]
    return [
        QueryEntityLink(
            entity_id=link.entity_id,
            name=link.name,
            score=link.score,
            rank=index + 1,
            matched_text=link.matched_text,
            source=link.source,
        )
        for index, link in enumerate(ordered)
    ]
