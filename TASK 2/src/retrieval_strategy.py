from __future__ import annotations

import math
import re
from collections import Counter

from .models import ContextItem, GraphArtifacts, QueryEntityLink, RetrievalPlan


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
CAPITALIZED_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?|[A-Z]{2,})"
    r"(?:\s+(?:[A-Z][A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?|[A-Z]{2,})){0,4}\b"
)
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


def build_retrieval_plan(
    question: str,
    question_type: str,
    top_k: int,
    relation_depth: int,
) -> RetrievalPlan:
    detected_type = question_type or infer_question_type(question)
    if detected_type == "comparison_query":
        return RetrievalPlan(
            question_type=detected_type,
            strategy="compare_sources_with_entity_bridges",
            semantic_top_k=top_k + 4,
            entity_top_k=10,
            relation_depth=max(2, relation_depth),
            evidence_budget=48,
            citation_budget=4,
            require_source_diversity=True,
            answer_mode="yes_no_or_contrast",
        )
    if detected_type == "temporal_query":
        return RetrievalPlan(
            question_type=detected_type,
            strategy="temporal_entity_evidence_paths",
            semantic_top_k=top_k + 4,
            entity_top_k=8,
            relation_depth=max(2, relation_depth + 1),
            evidence_budget=42,
            citation_budget=4,
            require_source_diversity=True,
            answer_mode="temporal_extract",
        )
    if detected_type == "null_query":
        return RetrievalPlan(
            question_type=detected_type,
            strategy="strict_evidence_or_insufficient",
            semantic_top_k=top_k,
            entity_top_k=6,
            relation_depth=1,
            evidence_budget=24,
            citation_budget=3,
            require_source_diversity=False,
            answer_mode="insufficient_guard",
        )
    if detected_type == "global_query":
        return RetrievalPlan(
            question_type=detected_type,
            strategy="global_community_summary",
            semantic_top_k=top_k + 12,
            entity_top_k=12,
            relation_depth=max(2, relation_depth),
            evidence_budget=64,
            citation_budget=6,
            require_source_diversity=True,
            answer_mode="comprehensive_summary",
        )
    return RetrievalPlan(
        question_type=detected_type,
        strategy="inference_entity_evidence_paths",
        semantic_top_k=top_k,
        entity_top_k=8,
        relation_depth=max(2, relation_depth),
        evidence_budget=36,
        citation_budget=3,
        require_source_diversity=True,
        answer_mode="entity_extract",
    )


def infer_question_type(question: str) -> str:
    lowered = question.lower()
    if lowered.startswith(("do ", "does ", "did ", "is ", "are ", "was ", "were ")):
        return "comparison_query"
    if any(word in lowered for word in ("before", "after", "during", "earlier", "later", "first", "when")):
        return "temporal_query"
    if any(word in lowered for word in ("summary", "summarize", "overall", "comprehensive", "general")):
        return "global_query"
    return "inference_query"


def rerank_context_items(
    question: str,
    context_items: list[ContextItem],
    entity_links: list[QueryEntityLink],
    plan: RetrievalPlan,
    artifacts: GraphArtifacts | None,
) -> tuple[list[ContextItem], str]:
    if artifacts is None:
        return context_items[: plan.evidence_budget], ""

    chunks_by_id = {chunk.chunk_id: chunk for chunk in artifacts.chunks}
    entities_by_id = {entity.entity_id: entity for entity in artifacts.entities}
    question_tokens = significant_tokens(question)
    linked_names = [link.name.lower() for link in entity_links]
    linked_entity_ids = {link.entity_id for link in entity_links}
    predicted_answer = predict_answer_candidate(question, context_items, artifacts, plan)
    predicted_lower = predicted_answer.lower()
    
    # Pre-calculate community sizes for normalization
    community_sizes = Counter(c.community for c in artifacts.chunks if c.community != -1)

    rescored: list[ContextItem] = []
    for item in context_items:
        if not item.node_id.startswith("chunk:") or item.node_id not in chunks_by_id:
            rescored.append(item)
            continue
        chunk = chunks_by_id[item.node_id]
        chunk_lower = chunk.text.lower()
        overlap = token_overlap(question_tokens, significant_tokens(chunk.text))
        entity_bonus = sum(1 for name in linked_names if name and name in chunk_lower)
        answer_bonus = 1.0 if predicted_lower and predicted_lower in chunk_lower else 0.0
        
        # Phase 2: Importance-based boosting using PageRank
        # Reduced from 10.0 to 0.1 to prevent generic hubs from dominating specific answers
        pagerank_bonus = chunk.pagerank * 0.1 
        
        # Phase 3: Context Coherence using Lynxes Communities (Normalized)
        coherence_bonus = 0.0
        if chunk.community != -1 and linked_entity_ids:
            shared_entities = sum(1 for eid in linked_entity_ids if eid in entities_by_id and entities_by_id[eid].community == chunk.community)
            if shared_entities > 0:
                # Use Logarithmic normalization to penalize giant communities
                # More shared entities = higher bonus; Larger community = lower density = lower bonus
                comm_size = community_sizes.get(chunk.community, 1)
                density_factor = 1.0 / math.log(2 + comm_size)
                coherence_bonus = 0.75 * (shared_entities / len(linked_entity_ids)) * density_factor
        
        score = item.score + 0.45 * overlap + 0.12 * entity_bonus + 0.35 * answer_bonus + pagerank_bonus + coherence_bonus
        rescored.append(
            ContextItem(
                node_id=item.node_id,
                node_type=item.node_type,
                score=score,
                reason=f"{item.reason}; answer-aware rerank (PR: {chunk.pagerank:.4f}, Comm: {chunk.community})",
                path=item.path,
            )
        )

    sorted_items = sorted(rescored, key=lambda value: (-value.score, value.node_id))
    if plan.require_source_diversity:
        sorted_items = diversify_by_document(sorted_items, artifacts)
    return sorted_items[: plan.evidence_budget], predicted_answer


def predict_answer_candidate(
    question: str,
    context_items: list[ContextItem],
    artifacts: GraphArtifacts,
    plan: RetrievalPlan,
) -> str:
    chunks_by_id = {chunk.chunk_id: chunk for chunk in artifacts.chunks}
    question_lower = question.lower()
    if plan.answer_mode == "insufficient_guard" and not context_items:
        return "Insufficient evidence"
    if plan.answer_mode == "yes_no_or_contrast" and question_lower.startswith(
        ("do ", "does ", "did ", "is ", "are ", "was ", "were ")
    ):
        return "Yes" if context_items else "Insufficient evidence"

    question_phrases = {phrase.lower() for phrase in capitalized_phrases(question)}
    weighted: Counter[str] = Counter()
    for item in context_items[:80]:
        if not item.node_id.startswith("chunk:") or item.node_id not in chunks_by_id:
            continue
        for phrase in capitalized_phrases(chunks_by_id[item.node_id].text):
            lower = phrase.lower()
            if lower in question_phrases:
                continue
            if len(lower) < 3:
                continue
            weighted[phrase] += max(0.01, item.score)
    if not weighted:
        return ""
    return weighted.most_common(1)[0][0]


def diversify_by_document(items: list[ContextItem], artifacts: GraphArtifacts) -> list[ContextItem]:
    chunk_to_document = {chunk.chunk_id: chunk.document_id for chunk in artifacts.chunks}
    first_pass: list[ContextItem] = []
    later: list[ContextItem] = []
    seen_documents: set[str] = set()
    for item in items:
        doc_id = chunk_to_document.get(item.node_id)
        if doc_id is None:
            later.append(item)
            continue
        if doc_id in seen_documents:
            later.append(item)
            continue
        seen_documents.add(doc_id)
        first_pass.append(item)
    return [*first_pass, *later]


def capitalized_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    for match in CAPITALIZED_PHRASE_RE.finditer(text):
        phrase = " ".join(match.group(0).split()).strip(" .,:;()[]{}\"'")
        if phrase and phrase.split()[0].lower() not in STOPWORDS:
            phrases.append(phrase)
    return phrases


def significant_tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text) if token.lower() not in STOPWORDS and len(token) > 2}


def token_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)
