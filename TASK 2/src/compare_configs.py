from __future__ import annotations

import csv
import inspect
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from .config import BenchmarkConfig
from .answer_generator import format_answer_log, generate_answer_and_citations
from .document_loader import load_benchmark_questions
from .evaluator import evaluate_runs
from .models import Answer, BenchmarkQuestion, BenchmarkResult, Citation, EvaluationResult, GraphArtifacts, RetrievalTrace
from .pipeline import build_artifacts
from .proposal import format_database_proposal
from .retriever import RetrievalRunResult, run_retrieval
from .storage.base import MissingExternalService, StorageAdapter
from .storage.caracal_adapter import CaracalStorageAdapter
from .storage.caracal_external_semantic_adapter import CaracalExternalSemanticAdapter
from .storage.neo4j_adapter import Neo4jStorageAdapter
from .storage.neo4j_external_semantic_adapter import Neo4jExternalSemanticAdapter


AdapterFactory = Callable[[BenchmarkConfig], StorageAdapter]


ADAPTERS: dict[str, AdapterFactory] = {
    "caracal-only": lambda config: CaracalStorageAdapter(config.caracal_db_path),
    "neo4j-only": lambda config: Neo4jStorageAdapter(),
    "caracal-external-semantic": lambda config: CaracalExternalSemanticAdapter(
        config.caracal_db_path,
        config.vector_store_dir,
    ),
    "neo4j-external-semantic": lambda config: Neo4jExternalSemanticAdapter(config.vector_store_dir),
}


def run_config_comparison(config: BenchmarkConfig) -> list[BenchmarkResult]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts = build_artifacts(config)
    questions = load_benchmark_questions(config)
    results: list[BenchmarkResult] = []
    traces: list[RetrievalTrace] = []
    answer_pairs: list[tuple[Answer, list[Citation]]] = []
    evaluations: list[EvaluationResult] = []

    for config_id in config.configs:
        factory = ADAPTERS[config_id]
        adapter = factory(config)
        try:
            result, adapter_traces, adapter_answers, adapter_evaluations = run_single_config(
                adapter,
                artifacts,
                questions,
                config,
            )
            results.append(result)
            traces.extend(adapter_traces)
            answer_pairs.extend(adapter_answers)
            evaluations.extend(adapter_evaluations)
        except MissingExternalService as exc:
            results.append(skipped_result(adapter, artifacts, str(exc)))
        finally:
            adapter.close()

    write_outputs(results, traces, answer_pairs, evaluations, config.output_dir)
    return results


def run_single_config(
    adapter: StorageAdapter,
    artifacts: GraphArtifacts,
    questions: list[BenchmarkQuestion],
    config: BenchmarkConfig,
) -> tuple[BenchmarkResult, list[RetrievalTrace], list[tuple[Answer, list[Citation]]], list[EvaluationResult]]:
    started = time.perf_counter()
    adapter.load(artifacts)
    load_seconds = time.perf_counter() - started

    run_results: list[RetrievalRunResult] = []
    for repeat in range(config.repeats):
        for question in questions:
            run_results.append(
                run_retrieval(
                    adapter=adapter,
                    question_id=question.question_id,
                    question=question.question,
                    embedding_dimension=config.embedding_dimension,
                    top_k=config.top_k_semantic_candidates,
                    relation_depth=config.relation_depth,
                    run_id=f"{adapter.config_id}:{question.question_id}:{repeat + 1}",
                    question_type=question.question_type,
                )
            )

    query_planning_seconds = median([run.timing.query_planning_seconds for run in run_results])
    semantic_entry_seconds = median([run.timing.semantic_entry_seconds for run in run_results])
    query_entity_linking_seconds = median([run.timing.query_entity_linking_seconds for run in run_results])
    semantic_reentry_seconds = median([run.timing.semantic_reentry_seconds for run in run_results])
    relation_expansion_seconds = median([run.timing.relation_expansion_seconds for run in run_results])
    answer_aware_reranking_seconds = median([run.timing.answer_aware_reranking_seconds for run in run_results])
    trace_write_seconds = median([run.timing.trace_write_seconds for run in run_results])
    total_seconds = load_seconds + sum(run.timing.total_seconds for run in run_results)
    traces = [run.trace for run in run_results]
    answer_pairs: list[tuple[Answer, list[Citation]]] = []
    for run in run_results:
        answer, citations = generate_answer_and_citations(run.trace, artifacts)
        adapter.store_answer(answer, citations)
        answer_pairs.append((answer, citations))
    evaluations = evaluate_runs(run_results, answer_pairs, artifacts, questions)
    counts = artifacts.counts()

    return (
        BenchmarkResult(
            config_id=adapter.config_id,
            config_name=adapter.config_name,
            graph_db=adapter.graph_db,
            vector_db=adapter.vector_db,
            status="ok",
            total_seconds=total_seconds,
            load_seconds=load_seconds,
            semantic_entry_seconds=semantic_entry_seconds,
            semantic_reentry_seconds=semantic_reentry_seconds,
            relation_expansion_seconds=relation_expansion_seconds,
            trace_write_seconds=trace_write_seconds,
            query_planning_seconds=query_planning_seconds,
            query_entity_linking_seconds=query_entity_linking_seconds,
            answer_aware_reranking_seconds=answer_aware_reranking_seconds,
            documents=counts["documents"],
            chunks=counts["chunks"],
            entities=counts["entities"],
            relationships=counts["relationships"],
            retrieved_context_items=sum(len(run.trace.context_items) for run in run_results),
            citations=sum(len(citations) for _answer, citations in answer_pairs),
            adapter_loc=adapter_loc(adapter.__class__),
            semantic_entry_mode=adapter.semantic_entry_mode,
            semantic_reentry_mode=adapter.semantic_reentry_mode,
            relation_expand_mode=adapter.relation_expand_mode,
            cross_store_join_count=adapter.cross_store_join_count,
            sync_notes=adapter.sync_notes,
            notes=adapter.notes,
        ),
        traces,
        answer_pairs,
        evaluations,
    )


def skipped_result(adapter: StorageAdapter, artifacts: GraphArtifacts, notes: str) -> BenchmarkResult:
    counts = artifacts.counts()
    return BenchmarkResult(
        config_id=adapter.config_id,
        config_name=adapter.config_name,
        graph_db=adapter.graph_db,
        vector_db=adapter.vector_db,
        status="skipped",
        total_seconds=None,
        load_seconds=None,
        semantic_entry_seconds=None,
        semantic_reentry_seconds=None,
        relation_expansion_seconds=None,
        trace_write_seconds=None,
        query_planning_seconds=None,
        query_entity_linking_seconds=None,
        answer_aware_reranking_seconds=None,
        documents=counts["documents"],
        chunks=counts["chunks"],
        entities=counts["entities"],
        relationships=counts["relationships"],
        retrieved_context_items=None,
        citations=None,
        adapter_loc=adapter_loc(adapter.__class__),
        semantic_entry_mode="not_run",
        semantic_reentry_mode="not_run",
        relation_expand_mode="not_run",
        cross_store_join_count=0,
        sync_notes="",
        notes=notes,
    )


from .quality_reporter import generate_quality_summary


def write_outputs(
    results: list[BenchmarkResult],
    traces: list[RetrievalTrace],
    answer_pairs: list[tuple[Answer, list[Citation]]],
    evaluations: list[EvaluationResult],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "comparison_benchmark.csv", [asdict(result) for result in results])
    write_csv(output_dir / "evaluation_report.csv", [asdict(result) for result in evaluations])
    generate_quality_summary(output_dir / "evaluation_report.csv", output_dir / "quality_summary.md")
    (output_dir / "comparison_report.txt").write_text(format_report(results), encoding="utf-8")
    (output_dir / "answer_log.md").write_text(format_answer_log(answer_pairs), encoding="utf-8")
    (output_dir / "database_proposal.md").write_text(format_database_proposal(results), encoding="utf-8")
    (output_dir / "retrieval_trace.json").write_text(
        json.dumps([trace.to_json_dict() for trace in traces], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_report(results: list[BenchmarkResult]) -> str:
    lines = [
        "# GraphRAG Configuration Comparison",
        "",
        "| Config | Architecture | Status | Total Seconds | Context Items | Semantic Entry | Relation Expand | Notes |",
        "| ------ | ------------ | ------ | ------------: | ------------: | -------------- | --------------- | ----- |",
    ]
    for result in results:
        total = "-" if result.total_seconds is None else f"{result.total_seconds:.6f}"
        contexts = "-" if result.retrieved_context_items is None else str(result.retrieved_context_items)
        lines.append(
            f"| {result.config_id} | {result.config_name} | {result.status} | {total} | {contexts} | "
            f"{result.semantic_entry_mode} | {result.relation_expand_mode} | {result.notes} |"
        )
    lines.append("")
    return "\n".join(lines)


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    return statistics.median(values)


def adapter_loc(cls: type[StorageAdapter]) -> int:
    try:
        return len(inspect.getsource(cls).splitlines())
    except OSError:
        return 0
