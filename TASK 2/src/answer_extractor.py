from __future__ import annotations

import re
from collections import Counter, defaultdict

from .models import AnswerCandidate, Chunk, ContextItem, GraphArtifacts, QueryEntityLink, RetrievalPlan
from .retrieval_strategy import capitalized_phrases, significant_tokens, token_overlap


DATE_RE = re.compile(
    r"\b(?:Jan\.?|January|Feb\.?|February|Mar\.?|March|Apr\.?|April|May|Jun\.?|June|Jul\.?|July|"
    r"Aug\.?|August|Sep\.?|Sept\.?|September|Oct\.?|October|Nov\.?|November|Dec\.?|December)"
    r"\s+\d{1,2}(?:,\s+\d{4})?\b|\b(?:19|20)\d{2}\b"
)

ORG_HINTS = {
    "ai",
    "bank",
    "bet",
    "bets",
    "book",
    "books",
    "capital",
    "company",
    "corp",
    "corporation",
    "exchange",
    "foundation",
    "fund",
    "google",
    "group",
    "inc",
    "labs",
    "llc",
    "openai",
    "organisation",
    "organization",
    "platform",
    "sportsbook",
    "technologies",
}

PLACE_HINTS = {"avenue", "manhattan", "park", "tower", "york"}
PERSON_HINTS = {"who", "individual", "person", "founder", "ceo", "executive", "figure"}
ORG_QUESTION_HINTS = {"company", "organization", "platform", "startup", "agency", "group"}
BAD_CANDIDATES = {
    "i",
    "it",
    "its",
    "he",
    "she",
    "they",
    "them",
    "this",
    "that",
    "these",
    "those",
    "we",
    "you",
    "your",
    "will",
    "bonus bet",
    "bonus bets",
    "how",
}
SOURCE_NAMES = {
    "ars technica",
    "associated press",
    "bbc",
    "bloomberg",
    "business insider",
    "cnn",
    "engadget",
    "fortune",
    "hacker news",
    "mashable",
    "new york times",
    "reuters",
    "sporting news",
    "techcrunch",
    "the verge",
    "the new york times",
    "washington post",
    "wired",
}


def extract_answer_from_context(
    question: str,
    context_items: list[ContextItem],
    entity_links: list[QueryEntityLink],
    plan: RetrievalPlan,
    artifacts: GraphArtifacts | None,
) -> tuple[str, list[AnswerCandidate]]:
    if artifacts is None:
        return "", []

    chunks_by_id = {chunk.chunk_id: chunk for chunk in artifacts.chunks}
    evidence_chunks = [
        chunks_by_id[item.node_id]
        for item in context_items
        if item.node_id.startswith("chunk:") and item.node_id in chunks_by_id
    ]
    if plan.answer_mode == "insufficient_guard":
        if not evidence_chunks:
            return "Insufficient evidence", []

    if plan.answer_mode == "yes_no_or_contrast":
        candidate = yes_no_candidate(question, evidence_chunks, plan)
        return candidate.candidate_text, [candidate]

    if plan.answer_mode == "temporal_extract":
        candidates = date_candidates(question, context_items, chunks_by_id)
        if candidates:
            return candidates[0].candidate_text, candidates

    candidates = entity_span_candidates(question, context_items, chunks_by_id, entity_links, plan)
    if not candidates:
        candidates = fallback_span_candidates(question, context_items, chunks_by_id)
    if not candidates:
        return "Insufficient evidence", []
    return candidates[0].candidate_text, candidates


def yes_no_candidate(question: str, evidence_chunks: list[Chunk], plan: RetrievalPlan) -> AnswerCandidate:
    question_tokens = significant_tokens(question)
    chunk_tokens = [significant_tokens(chunk.text) for chunk in evidence_chunks[: plan.evidence_budget]]
    supporting_chunks = [
        chunk
        for chunk, tokens in zip(evidence_chunks, chunk_tokens, strict=False)
        if token_overlap(question_tokens, tokens) > 0.08
    ]
    if not supporting_chunks:
        supporting_chunks = evidence_chunks[: min(2, len(evidence_chunks))]
    negative_markers = {"not", "never", "no", "different", "unlike", "however", "but"}
    negative_hits = sum(1 for chunk in supporting_chunks if negative_markers & significant_tokens(chunk.text))
    answer = "No" if negative_hits >= max(2, len(supporting_chunks)) else "Yes"
    documents = unique_tuple(chunk.document_id for chunk in supporting_chunks)
    return AnswerCandidate(
        candidate_text=answer,
        candidate_type="yes_no",
        score=1.0 + 0.1 * len(documents),
        rank=1,
        source_chunk_ids=unique_tuple(chunk.chunk_id for chunk in supporting_chunks),
        source_document_ids=documents,
        frequency=len(supporting_chunks),
        reason="question type is comparison_query; evidence supports binary answer",
    )


def date_candidates(
    question: str,
    context_items: list[ContextItem],
    chunks_by_id: dict[str, Chunk],
) -> list[AnswerCandidate]:
    question_text = question.lower()
    candidate_rows: dict[str, CandidateAccumulator] = {}
    for position, item in enumerate(context_items):
        if item.node_id not in chunks_by_id:
            continue
        chunk = chunks_by_id[item.node_id]
        for match in DATE_RE.finditer(chunk.text):
            text = normalize_candidate(match.group(0))
            if not text or text.lower() in question_text:
                continue
            accumulator = candidate_rows.setdefault(text, CandidateAccumulator(text, "date"))
            accumulator.add(chunk, item.score, 1.0 / (position + 1), "date expression in retrieved evidence")
    return finalize_candidates(candidate_rows.values())


def entity_span_candidates(
    question: str,
    context_items: list[ContextItem],
    chunks_by_id: dict[str, Chunk],
    entity_links: list[QueryEntityLink],
    plan: RetrievalPlan,
) -> list[AnswerCandidate]:
    question_lower = question.lower()
    question_tokens = significant_tokens(question)
    linked_names = {link.name.lower() for link in entity_links}
    preferred_type = preferred_candidate_type(question, plan)
    candidate_rows: dict[str, CandidateAccumulator] = {}

    for position, item in enumerate(context_items):
        if item.node_id not in chunks_by_id:
            continue
        chunk = chunks_by_id[item.node_id]
        chunk_tokens = significant_tokens(chunk.text)
        overlap = token_overlap(question_tokens, chunk_tokens)
        for phrase in capitalized_phrases(chunk.text):
            text = normalize_candidate(phrase)
            if not is_valid_answer_phrase(text, question_lower):
                continue
            candidate_type = classify_candidate(text)
            accumulator = candidate_rows.setdefault(text, CandidateAccumulator(text, candidate_type))
            type_bonus = type_fit_bonus(text, candidate_type, preferred_type)
            linked_penalty = -0.35 if text.lower() in linked_names else 0.0
            question_penalty = -0.45 if text.lower() in question_lower else 0.0
            source_bonus = 0.12 if len(item.path) >= 3 else 0.0
            rank_bonus = 1.0 / (position + 1)
            score = item.score + 0.7 * overlap + type_bonus + linked_penalty + question_penalty + source_bonus
            accumulator.add(chunk, score, rank_bonus, f"{candidate_type} span in retrieved evidence")

    enhance_full_name_candidates(candidate_rows, preferred_type)
    return finalize_candidates(candidate_rows.values())


def fallback_span_candidates(
    question: str,
    context_items: list[ContextItem],
    chunks_by_id: dict[str, Chunk],
) -> list[AnswerCandidate]:
    question_tokens = significant_tokens(question)
    weighted: Counter[str] = Counter()
    sources: dict[str, set[str]] = defaultdict(set)
    chunks: dict[str, set[str]] = defaultdict(set)
    for item in context_items:
        if item.node_id not in chunks_by_id:
            continue
        chunk = chunks_by_id[item.node_id]
        for token in significant_tokens(chunk.text):
            if token in question_tokens or len(token) < 4:
                continue
            weighted[token] += max(0.01, item.score)
            sources[token].add(chunk.document_id)
            chunks[token].add(chunk.chunk_id)
    candidates: list[AnswerCandidate] = []
    for rank, (token, score) in enumerate(weighted.most_common(10), start=1):
        candidates.append(
            AnswerCandidate(
                candidate_text=token,
                candidate_type="token",
                score=float(score),
                rank=rank,
                source_chunk_ids=tuple(sorted(chunks[token])),
                source_document_ids=tuple(sorted(sources[token])),
                frequency=len(chunks[token]),
                reason="fallback token extracted from retrieved evidence",
            )
        )
    return candidates


class CandidateAccumulator:
    def __init__(self, text: str, candidate_type: str) -> None:
        self.text = text
        self.candidate_type = candidate_type
        self.score = 0.0
        self.frequency = 0
        self.chunk_ids: set[str] = set()
        self.document_ids: set[str] = set()
        self.reasons: Counter[str] = Counter()

    def add(self, chunk: Chunk, score: float, rank_bonus: float, reason: str) -> None:
        self.score += max(0.0, score) + 0.15 * rank_bonus
        self.frequency += 1
        self.chunk_ids.add(chunk.chunk_id)
        self.document_ids.add(chunk.document_id)
        self.reasons[reason] += 1

    def to_candidate(self, rank: int) -> AnswerCandidate:
        source_diversity = 0.3 * len(self.document_ids)
        frequency_bonus = 0.08 * self.frequency
        return AnswerCandidate(
            candidate_text=self.text,
            candidate_type=self.candidate_type,
            score=float(self.score + source_diversity + frequency_bonus),
            rank=rank,
            source_chunk_ids=tuple(sorted(self.chunk_ids)),
            source_document_ids=tuple(sorted(self.document_ids)),
            frequency=self.frequency,
            reason=self.reasons.most_common(1)[0][0] if self.reasons else "evidence span",
        )


def finalize_candidates(accumulators) -> list[AnswerCandidate]:
    rough = [accumulator.to_candidate(0) for accumulator in accumulators]
    rough.sort(key=lambda candidate: (-candidate.score, -len(candidate.source_document_ids), candidate.candidate_text))
    return [
        AnswerCandidate(
            candidate_text=candidate.candidate_text,
            candidate_type=candidate.candidate_type,
            score=candidate.score,
            rank=index + 1,
            source_chunk_ids=candidate.source_chunk_ids,
            source_document_ids=candidate.source_document_ids,
            frequency=candidate.frequency,
            reason=candidate.reason,
        )
        for index, candidate in enumerate(rough[:20])
    ]


def preferred_candidate_type(question: str, plan: RetrievalPlan) -> str:
    tokens = significant_tokens(question)
    if plan.answer_mode == "temporal_extract":
        return "date"
    if tokens & PERSON_HINTS:
        return "person"
    if tokens & ORG_QUESTION_HINTS:
        return "organization"
    return "entity"


def classify_candidate(text: str) -> str:
    lowered = text.lower()
    words = lowered.split()
    if DATE_RE.fullmatch(text):
        return "date"
    if lowered in {"new york"} or any(hint in words for hint in PLACE_HINTS):
        return "place"
    if words and (words[-1].strip(".") in ORG_HINTS or any(hint in words for hint in ORG_HINTS)):
        return "organization"
    if len(text.split()) >= 2 or "-" in text:
        return "person"
    return "entity"


def type_fit_bonus(text: str, candidate_type: str, preferred_type: str) -> float:
    if candidate_type == preferred_type:
        return 0.65
    if preferred_type == "person" and candidate_type == "entity" and "-" in text:
        return 0.35
    if preferred_type == "person" and candidate_type in {"organization", "place"}:
        return -0.95
    if preferred_type == "person" and candidate_type == "entity":
        return -0.85
    if preferred_type == "organization" and candidate_type == "place":
        return -0.65
    if preferred_type == "organization" and candidate_type == "entity":
        return -0.15
    if preferred_type == "organization" and candidate_type == "person":
        return -0.35
    return 0.0


def enhance_full_name_candidates(candidate_rows: dict[str, CandidateAccumulator], preferred_type: str) -> None:
    if preferred_type != "person":
        return
    rows = list(candidate_rows.values())
    for shorter in rows:
        short_key = shorter.text.lower()
        if len(shorter.text.split()) > 1 and "-" not in shorter.text:
            continue
        for longer in rows:
            long_key = longer.text.lower()
            if longer is shorter or len(longer.text) <= len(shorter.text):
                continue
            if short_key not in long_key:
                continue
            if len(longer.text.split()) < 2:
                continue
            transfer = 0.65 * shorter.score
            longer.score += transfer
            longer.frequency += max(0, shorter.frequency // 2)
            longer.chunk_ids.update(shorter.chunk_ids)
            longer.document_ids.update(shorter.document_ids)
            longer.reasons["full-name support from shorter mention"] += 1
            shorter.score *= 0.45


def is_valid_answer_phrase(text: str, question_lower: str) -> bool:
    lowered = text.lower()
    if len(text) < 2:
        return False
    if lowered in BAD_CANDIDATES:
        return False
    words = lowered.split()
    if len(words) != len(set(words)):
        return False
    if lowered in SOURCE_NAMES:
        return False
    if lowered.startswith(("title ", "source ", "category ", "author ", "published ")):
        return False
    if lowered in {"yes", "no"}:
        return False
    if lowered in question_lower and len(text.split()) <= 1:
        return False
    if len(text.split()) > 5:
        return False
    return True


def normalize_candidate(text: str) -> str:
    return " ".join(text.strip(" .,:;()[]{}\"'").split())


def unique_tuple(values) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)
