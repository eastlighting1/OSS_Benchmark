from __future__ import annotations

from src.config import BenchmarkConfig
from src.answer_extractor import extract_answer_from_context
from src.answer_generator import generate_answer_and_citations
from src.embeddings import cosine_similarity, embed_text
from src.evaluator import evaluate_runs
from src.models import BenchmarkQuestion, Chunk, ContextItem, GraphArtifacts, RetrievalPlan
from src.pipeline import build_artifacts
from src.retrieval_strategy import build_retrieval_plan
from src.retriever import run_retrieval
from src.storage.base import StorageAdapter


def test_build_artifacts_from_sample_documents() -> None:
    artifacts = build_artifacts(BenchmarkConfig())

    assert artifacts.documents
    assert artifacts.chunks
    assert artifacts.entities
    assert artifacts.relationships
    assert {record.owner_type for record in artifacts.embeddings} == {"Chunk", "Entity"}


def test_deterministic_embeddings_are_normalized() -> None:
    first = embed_text("GraphRAG semantic neighborhood", 32)
    second = embed_text("GraphRAG semantic neighborhood", 32)

    assert first == second
    assert abs(cosine_similarity(first, second) - 1.0) < 1e-9


def test_answer_generation_produces_citations() -> None:
    config = BenchmarkConfig()
    artifacts = build_artifacts(config)
    adapter = StorageAdapter()
    adapter.load(artifacts)

    run = run_retrieval(
        adapter=adapter,
        question_id="q_test",
        question="How do semantic neighborhoods support GraphRAG?",
        embedding_dimension=config.embedding_dimension,
        top_k=3,
        relation_depth=2,
        run_id="test-answer",
    )
    answer, citations = generate_answer_and_citations(run.trace, artifacts)

    assert answer.grounding_score > 0
    assert citations
    assert all(citation.chunk_id.startswith("chunk:") for citation in citations)


def test_question_type_builds_different_retrieval_plan() -> None:
    comparison = build_retrieval_plan("Do the two articles agree?", "comparison_query", top_k=3, relation_depth=1)
    inference = build_retrieval_plan("Who founded the company?", "inference_query", top_k=3, relation_depth=1)

    assert comparison.strategy != inference.strategy
    assert comparison.require_source_diversity
    assert comparison.citation_budget >= inference.citation_budget


def test_answer_correctness_metrics_use_predicted_answer() -> None:
    config = BenchmarkConfig()
    artifacts = build_artifacts(config)
    adapter = StorageAdapter()
    adapter.load(artifacts)

    run = run_retrieval(
        adapter=adapter,
        question_id="q_gold",
        question="Which database is proposed for GraphRAG?",
        embedding_dimension=config.embedding_dimension,
        top_k=3,
        relation_depth=2,
        run_id="test-correctness",
    )
    object.__setattr__(run.trace, "predicted_answer", "CaracalDB")
    answer, citations = generate_answer_and_citations(run.trace, artifacts)
    evaluations = evaluate_runs(
        [run],
        [(answer, citations)],
        artifacts,
        [BenchmarkQuestion(question_id="q_gold", question=run.trace.question, answer="CaracalDB")],
    )

    assert evaluations[0].answer_exact_match == 1.0
    assert evaluations[0].answer_token_f1 == 1.0


def test_answer_extractor_uses_retrieved_evidence_span() -> None:
    chunk = Chunk(
        chunk_id="chunk:evidence:001",
        document_id="doc:evidence",
        chunk_index=0,
        text="The highly anticipated criminal trial for Sam Bankman-Fried focused on fraud and conspiracy.",
        token_count=12,
    )
    artifacts = GraphArtifacts(
        documents=[],
        chunks=[chunk],
        entities=[],
        mentions=[],
        relationships=[],
        embeddings=[],
    )
    plan = RetrievalPlan(
        question_type="inference_query",
        strategy="inference_entity_evidence_paths",
        semantic_top_k=3,
        entity_top_k=3,
        relation_depth=2,
        evidence_budget=5,
        citation_budget=3,
        require_source_diversity=True,
        answer_mode="entity_extract",
    )

    answer, candidates = extract_answer_from_context(
        question="Who is facing a criminal trial for fraud and conspiracy?",
        context_items=[ContextItem("chunk:evidence:001", "Chunk", 1.0, "test evidence", ["chunk:evidence:001"])],
        entity_links=[],
        plan=plan,
        artifacts=artifacts,
    )

    assert answer == "Sam Bankman-Fried"
    assert candidates[0].source_chunk_ids == ("chunk:evidence:001",)
