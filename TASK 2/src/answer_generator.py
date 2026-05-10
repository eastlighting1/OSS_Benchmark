from __future__ import annotations

from .models import Answer, Citation, Chunk, ContextItem, GraphArtifacts, RetrievalPlan, RetrievalTrace
from .retrieval_strategy import significant_tokens, token_overlap


from functools import lru_cache

@lru_cache(maxsize=10000)
def significant_tokens_cached(text: str) -> frozenset[str]:
    return frozenset(significant_tokens(text))

def generate_answer_and_citations(
    trace: RetrievalTrace,
    artifacts: GraphArtifacts,
    max_citations: int = 3,
) -> tuple[Answer, list[Citation]]:
    chunks_by_id = {chunk.chunk_id: chunk for chunk in artifacts.chunks}
    citation_budget = trace.retrieval_plan.citation_budget if trace.retrieval_plan else max_citations
    cited_chunks = select_citation_chunks(
        question=trace.question,
        predicted_answer=trace.predicted_answer,
        context_items=trace.context_items,
        chunks_by_id=chunks_by_id,
        plan=trace.retrieval_plan,
        max_citations=citation_budget,
    )

    predicted = trace.predicted_answer.strip()
    if cited_chunks:
        joined = " ".join(short_sentence(chunk.text) for chunk in cited_chunks)
        if predicted:
            answer_text = f"Predicted answer: {predicted}. Evidence summary: {joined}"
        else:
            answer_text = (
                f"The retrieved graph ecosystem context indicates that {joined} "
                f"This answer is grounded in {len(cited_chunks)} cited chunk(s)."
            )
    else:
        if predicted and predicted.lower().startswith("insufficient"):
            answer_text = predicted
        else:
            answer_text = "The retrieved context is insufficient to answer this question with citations."

    grounding_score = min(1.0, len(cited_chunks) / max(1, citation_budget))
    answer = Answer(
        answer_id=f"answer:{trace.run_id}",
        question_id=trace.question_id,
        question=trace.question,
        answer_text=answer_text,
        grounding_score=grounding_score,
        config_id=trace.config_id,
    )
    citations = [
        Citation(
            citation_id=f"citation:{trace.run_id}:{index + 1}",
            answer_id=answer.answer_id,
            chunk_id=chunk.chunk_id,
            evidence_text=chunk.text[:240],
            confidence=citation_confidence(trace.question, trace.predicted_answer, chunk),
        )
        for index, chunk in enumerate(cited_chunks)
    ]
    return answer, citations


def select_citation_chunks(
    question: str,
    predicted_answer: str,
    context_items: list[ContextItem],
    chunks_by_id: dict[str, Chunk],
    plan: RetrievalPlan | None,
    max_citations: int,
) -> list[Chunk]:
    scored: list[tuple[float, int, Chunk]] = []
    question_tokens = significant_tokens_cached(question)
    answer_lower = predicted_answer.lower().strip()
    for position, item in enumerate(context_items):
        if not item.node_id.startswith("chunk:") or item.node_id not in chunks_by_id:
            continue
        chunk = chunks_by_id[item.node_id]
        chunk_lower = chunk.text.lower()
        overlap = token_overlap(question_tokens, significant_tokens_cached(chunk.text))
        answer_bonus = 0.5 if answer_lower and answer_lower in chunk_lower else 0.0
        path_bonus = 0.12 if len(item.path) >= 3 else 0.0
        score = item.score + 0.55 * overlap + answer_bonus + path_bonus
        scored.append((score, position, chunk))

    scored.sort(key=lambda row: (-row[0], row[1], row[2].chunk_id))
    if plan is not None and plan.require_source_diversity:
        return diversify_citation_chunks(scored, max_citations)
    return [chunk for _score, _position, chunk in scored[:max_citations]]


def diversify_citation_chunks(scored: list[tuple[float, int, Chunk]], max_citations: int) -> list[Chunk]:
    selected: list[Chunk] = []
    seen_documents: set[str] = set()
    for _score, _position, chunk in scored:
        if chunk.document_id in seen_documents:
            continue
        selected.append(chunk)
        seen_documents.add(chunk.document_id)
        if len(selected) >= max_citations:
            return selected
    for _score, _position, chunk in scored:
        if chunk in selected:
            continue
        selected.append(chunk)
        if len(selected) >= max_citations:
            return selected
    return selected


def citation_confidence(question: str, predicted_answer: str, chunk: Chunk) -> float:
    overlap = token_overlap(significant_tokens_cached(question), significant_tokens_cached(chunk.text))
    answer_bonus = 0.25 if predicted_answer and predicted_answer.lower() in chunk.text.lower() else 0.0
    return min(1.0, 0.5 + 0.5 * overlap + answer_bonus)


def short_sentence(text: str, max_chars: int = 220) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


def format_answer_log(pairs: list[tuple[Answer, list[Citation]]]) -> str:
    lines = ["# GraphRAG Answer Log", ""]
    for answer, citations in pairs:
        lines.extend(
            [
                f"## {answer.answer_id}",
                "",
                f"Config: `{answer.config_id}`",
                "",
                f"Question: {answer.question}",
                "",
                answer.answer_text,
                "",
                "Citations:",
            ]
        )
        if citations:
            for citation in citations:
                lines.append(f"- `{citation.chunk_id}` confidence={citation.confidence:.2f}: {citation.evidence_text}")
        else:
            lines.append("- No citations.")
        lines.append("")
    return "\n".join(lines)
