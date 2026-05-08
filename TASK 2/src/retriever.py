from __future__ import annotations

import time
from dataclasses import dataclass

from .answer_extractor import extract_answer_from_context
from .embeddings import embed_text
from .models import ContextItem, RetrievalTrace, SemanticCandidate
from .retrieval_strategy import build_retrieval_plan, rerank_context_items
from .storage.base import StorageAdapter


@dataclass(frozen=True)
class RetrievalTiming:
    query_planning_seconds: float
    semantic_entry_seconds: float
    query_entity_linking_seconds: float
    semantic_reentry_seconds: float
    relation_expansion_seconds: float
    answer_aware_reranking_seconds: float
    trace_write_seconds: float

    @property
    def total_seconds(self) -> float:
        return (
            self.query_planning_seconds
            + self.semantic_entry_seconds
            + self.query_entity_linking_seconds
            + self.semantic_reentry_seconds
            + self.relation_expansion_seconds
            + self.answer_aware_reranking_seconds
            + self.trace_write_seconds
        )


@dataclass(frozen=True)
class RetrievalRunResult:
    trace: RetrievalTrace
    timing: RetrievalTiming


def run_retrieval(
    adapter: StorageAdapter,
    question_id: str,
    question: str,
    embedding_dimension: int,
    top_k: int,
    relation_depth: int,
    run_id: str,
    question_type: str = "",
) -> RetrievalRunResult:
    started = time.perf_counter()
    plan = build_retrieval_plan(question, question_type, top_k, relation_depth)
    query_planning_seconds = time.perf_counter() - started
    query_embedding = embed_text(question, embedding_dimension)

    native_retrieval = adapter.native_graph_retrieval(question, query_embedding, plan)
    if native_retrieval is not None:
        semantic_candidates = native_retrieval.semantic_candidates
        query_entity_links = native_retrieval.query_entity_links
        context_items = native_retrieval.context_items
        operator_timings = native_retrieval.operator_timings_ms
        semantic_entry_seconds = float(operator_timings.get("vector_graph_search", 0.0)) / 1000.0
        query_entity_linking_seconds = float(operator_timings.get("link_entities", 0.0)) / 1000.0
        relation_expansion_seconds = float(operator_timings.get("evidence_search", 0.0)) / 1000.0
        semantic_reentry_seconds = max(
            0.0,
            float(native_retrieval.profile.get("elapsed_ms", 0.0)) / 1000.0
            - semantic_entry_seconds
            - query_entity_linking_seconds
            - relation_expansion_seconds,
        )
    else:
        started = time.perf_counter()
        semantic_candidates = adapter.semantic_entry(question, query_embedding, plan.semantic_top_k)
        semantic_entry_seconds = time.perf_counter() - started

        started = time.perf_counter()
        query_entity_links = adapter.link_query_entities(question, query_embedding, plan.entity_top_k)
        query_entity_linking_seconds = time.perf_counter() - started

        started = time.perf_counter()
        graph_candidates = adapter.semantic_reentry(semantic_candidates)
        semantic_reentry_seconds = time.perf_counter() - started

        started = time.perf_counter()
        context_items = merge_context_items(
            [
                *semantic_candidates_as_context(graph_candidates),
                *entity_links_as_context(query_entity_links),
                *adapter.evidence_path_expand(graph_candidates, query_entity_links, plan),
            ]
        )
        relation_expansion_seconds = time.perf_counter() - started

    started = time.perf_counter()
    context_items, rerank_answer_hint = rerank_context_items(
        question=question,
        context_items=context_items,
        entity_links=query_entity_links,
        plan=plan,
        artifacts=adapter.artifacts,
    )
    predicted_answer, answer_candidates = extract_answer_from_context(
        question=question,
        context_items=context_items,
        entity_links=query_entity_links,
        plan=plan,
        artifacts=adapter.artifacts,
    )
    if not predicted_answer:
        predicted_answer = rerank_answer_hint
    answer_aware_reranking_seconds = time.perf_counter() - started

    evidence_paths = adapter.build_evidence_paths(context_items)
    trace = RetrievalTrace(
        run_id=run_id,
        question_id=question_id,
        question=question,
        config_id=adapter.config_id,
        semantic_entry_mode=adapter.semantic_entry_mode,
        semantic_reentry_mode=adapter.semantic_reentry_mode,
        relation_expand_mode=adapter.relation_expand_mode,
        semantic_candidates=semantic_candidates,
        context_items=context_items,
        evidence_paths=evidence_paths,
        query_entity_links=query_entity_links,
        retrieval_plan=plan,
        predicted_answer=predicted_answer,
        answer_candidates=answer_candidates,
    )

    started = time.perf_counter()
    adapter.store_retrieval_trace(trace)
    trace_write_seconds = time.perf_counter() - started

    return RetrievalRunResult(
        trace=trace,
        timing=RetrievalTiming(
            query_planning_seconds=query_planning_seconds,
            semantic_entry_seconds=semantic_entry_seconds,
            query_entity_linking_seconds=query_entity_linking_seconds,
            semantic_reentry_seconds=semantic_reentry_seconds,
            relation_expansion_seconds=relation_expansion_seconds,
            answer_aware_reranking_seconds=answer_aware_reranking_seconds,
            trace_write_seconds=trace_write_seconds,
        ),
    )


def semantic_candidates_as_context(candidates: list[SemanticCandidate]) -> list[ContextItem]:
    return [
        ContextItem(
            node_id=candidate.node_id,
            node_type=candidate.node_type,
            score=candidate.score,
            reason=f"semantic entry: {candidate.reason}",
            path=[candidate.node_id],
        )
        for candidate in candidates
    ]


def entity_links_as_context(entity_links) -> list[ContextItem]:
    return [
        ContextItem(
            node_id=link.entity_id,
            node_type="Entity",
            score=link.score,
            reason=f"query entity link: {link.source}",
            path=[link.entity_id],
        )
        for link in entity_links
    ]


def merge_context_items(items: list[ContextItem]) -> list[ContextItem]:
    merged: dict[str, ContextItem] = {}
    for item in items:
        previous = merged.get(item.node_id)
        if previous is None or item.score > previous.score:
            merged[item.node_id] = item
    return sorted(merged.values(), key=lambda item: (-item.score, item.node_id))
