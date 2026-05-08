from __future__ import annotations

import re

from .models import Answer, BenchmarkQuestion, Citation, EvaluationResult, GraphArtifacts
from .retriever import RetrievalRunResult


def evaluate_runs(
    run_results: list[RetrievalRunResult],
    answer_pairs: list[tuple[Answer, list[Citation]]],
    artifacts: GraphArtifacts | None = None,
    questions: list[BenchmarkQuestion] | None = None,
) -> list[EvaluationResult]:
    by_answer_id = {answer.answer_id: (answer, citations) for answer, citations in answer_pairs}
    questions_by_id = {question.question_id: question for question in questions or []}
    chunk_to_document = {chunk.chunk_id: chunk.document_id for chunk in artifacts.chunks} if artifacts else {}
    evaluations: list[EvaluationResult] = []
    for run in run_results:
        trace = run.trace
        answer_id = f"answer:{trace.run_id}"
        answer, citations = by_answer_id[answer_id]
        chunk_context_ids = [item.node_id for item in trace.context_items if item.node_id.startswith("chunk:")]
        chunk_context_count = len(chunk_context_ids)
        question = questions_by_id.get(trace.question_id)
        gold_documents = set(question.gold_document_ids) if question else set()
        if gold_documents and chunk_to_document:
            retrieved_documents = {chunk_to_document[chunk_id] for chunk_id in chunk_context_ids if chunk_id in chunk_to_document}
            relevant_chunks = [
                chunk_id
                for chunk_id in chunk_context_ids
                if chunk_to_document.get(chunk_id) in gold_documents
            ]
            retrieved_gold_documents = retrieved_documents & gold_documents
            cited_documents = {
                chunk_to_document[citation.chunk_id]
                for citation in citations
                if citation.chunk_id in chunk_to_document
            }
            cited_gold_documents = cited_documents & gold_documents
            precision = len(relevant_chunks) / max(1, chunk_context_count)
            context_recall = len(retrieved_gold_documents) / max(1, len(gold_documents))
            citation_coverage = len(cited_gold_documents) / max(1, len(gold_documents))
            gold_count = len(gold_documents)
            retrieved_gold_count = len(retrieved_gold_documents)
            evidence_recall = context_recall
            citation_recall = citation_coverage
        else:
            precision = chunk_context_count / max(1, len(trace.context_items))
            citation_coverage = len(citations) / max(1, chunk_context_count)
            context_recall = 1.0 if citations else 0.0
            gold_count = 0
            retrieved_gold_count = 0
            evidence_recall = 0.0
            citation_recall = 0.0
        gold_answer = question.answer if question else ""
        predicted_answer = trace.predicted_answer or answer.answer_text
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
                evidence_recall_at_context=evidence_recall,
                citation_recall=citation_recall,
                answer_exact_match=answer_exact_match,
                answer_contains_gold=answer_contains_gold,
                answer_token_f1=answer_token_f1,
            )
        )
    return evaluations


ANSWER_TOKEN_RE = re.compile(r"[a-z0-9]+")
ARTICLES_RE = re.compile(r"\b(a|an|the)\b")


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    without_articles = ARTICLES_RE.sub(" ", lowered)
    return " ".join(ANSWER_TOKEN_RE.findall(without_articles))


def exact_match(predicted: str, gold: str) -> float:
    return 1.0 if normalize_answer(predicted) == normalize_answer(gold) else 0.0


def contains_answer(predicted: str, gold: str) -> float:
    normalized_predicted = normalize_answer(predicted)
    normalized_gold = normalize_answer(gold)
    if not normalized_gold:
        return 0.0
    return 1.0 if normalized_gold in normalized_predicted else 0.0


def token_f1(predicted: str, gold: str) -> float:
    predicted_tokens = normalize_answer(predicted).split()
    gold_tokens = normalize_answer(gold).split()
    if not predicted_tokens or not gold_tokens:
        return 0.0
    common = 0
    remaining = predicted_tokens.copy()
    for token in gold_tokens:
        if token in remaining:
            common += 1
            remaining.remove(token)
    if common == 0:
        return 0.0
    precision = common / len(predicted_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
