from __future__ import annotations

import re
import polars as pl
import numpy as np
from functools import lru_cache

from .models import Answer, BenchmarkQuestion, Citation, EvaluationResult, GraphArtifacts
from .retriever import RetrievalRunResult


@lru_cache(maxsize=10000)
def normalize_answer_cached(text: str) -> str:
    lowered = text.lower()
    without_articles = ARTICLES_RE.sub(" ", lowered)
    return " ".join(ANSWER_TOKEN_RE.findall(without_articles))


def evaluate_runs(
    run_results: list[RetrievalRunResult],
    answer_pairs: list[tuple[Answer, list[Citation]]],
    artifacts: GraphArtifacts | None = None,
    questions: list[BenchmarkQuestion] | None = None,
) -> list[EvaluationResult]:
    # Faster lookup for small datasets
    questions_by_id = {question.question_id: question for question in questions or []}
    chunk_to_document = {chunk.chunk_id: chunk.document_id for chunk in artifacts.chunks} if artifacts else {}
    
    # Process batch for heavy NLP metrics
    evaluations: list[EvaluationResult] = []
    
    for run, (answer, citations) in zip(run_results, answer_pairs):
        trace = run.trace
        chunk_context_ids = [item.node_id for item in trace.context_items if item.node_id.startswith("chunk:")]
        chunk_context_count = len(chunk_context_ids)
        question = questions_by_id.get(trace.question_id)
        gold_documents = set(question.gold_document_ids) if question else set()
        
        if gold_documents and chunk_to_document:
            retrieved_documents = {chunk_to_document[chunk_id] for chunk_id in chunk_context_ids if chunk_id in chunk_to_document}
            relevant_chunks = [
                chunk_id for chunk_id in chunk_context_ids
                if chunk_to_document.get(chunk_id) in gold_documents
            ]
            retrieved_gold_documents = retrieved_documents & gold_documents
            cited_documents = {
                chunk_to_document[citation.chunk_id]
                for citation in citations if citation.chunk_id in chunk_to_document
            }
            cited_gold_documents = cited_documents & gold_documents
            
            precision = len(relevant_chunks) / max(1, chunk_context_count)
            context_recall = len(retrieved_gold_documents) / max(1, len(gold_documents))
            citation_coverage = len(cited_gold_documents) / max(1, len(gold_documents))
            gold_count = len(gold_documents)
            retrieved_gold_count = len(retrieved_gold_documents)
        else:
            precision = chunk_context_count / max(1, len(trace.context_items))
            citation_coverage = len(citations) / max(1, chunk_context_count)
            context_recall = 1.0 if citations else 0.0
            gold_count = 0
            retrieved_gold_count = 0

        gold_answer = question.answer if question else ""
        predicted_answer = trace.predicted_answer or answer.answer_text
        
        # Use cached normalization
        answer_exact_match = exact_match(predicted_answer, gold_answer) if gold_answer else 0.0
        answer_contains_gold = contains_answer(predicted_answer, gold_answer) if gold_answer else 0.0
        answer_token_f1 = token_f1(predicted_answer, gold_answer) if gold_answer else 0.0
        
        evaluations.append(
            EvaluationResult(
                config_id=trace.config_id,
                question_id=trace.question_id,
                retrieval_precision_at_k=precision,
                context_recall=context_recall,
                citation_coverage=min(1.0, citation_coverage),
                answer_grounding_score=answer.grounding_score,
                context_items=len(trace.context_items),
                citations=len(citations),
                latency_seconds=run.timing.total_seconds,
                gold_evidence_documents=gold_count,
                retrieved_gold_documents=retrieved_gold_count,
                evidence_recall_at_context=context_recall,
                citation_recall=citation_coverage,
                answer_exact_match=answer_exact_match,
                answer_contains_gold=answer_contains_gold,
                answer_token_f1=answer_token_f1,
            )
        )
    return evaluations


ANSWER_TOKEN_RE = re.compile(r"[a-z0-9]+")
ARTICLES_RE = re.compile(r"\b(a|an|the)\b")


def normalize_answer(text: str) -> str:
    return normalize_answer_cached(text)


def exact_match(predicted: str, gold: str) -> float:
    return 1.0 if normalize_answer(predicted) == normalize_answer(gold) else 0.0


def contains_answer(predicted: str, gold: str) -> float:
    normalized_predicted = normalize_answer(predicted)
    normalized_gold = normalize_answer(gold)
    if not normalized_gold:
        return 0.0
    return 1.0 if normalized_gold in normalized_predicted else 0.0


def token_f1(predicted: str, gold: str) -> float:
    p_norm = normalize_answer(predicted)
    g_norm = normalize_answer(gold)
    
    predicted_tokens = p_norm.split()
    gold_tokens = g_norm.split()
    
    if not predicted_tokens or not gold_tokens:
        return 0.0
    
    common = Counter(predicted_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(predicted_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

from collections import Counter
