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


from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

def run_config_comparison(config: BenchmarkConfig) -> list[BenchmarkResult]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize streaming output files
    trace_path = config.output_dir / "retrieval_trace.jsonl"
    answer_path = config.output_dir / "answer_log.md"
    # Clear previous runs
    trace_path.write_text("", encoding="utf-8")
    answer_path.write_text("# GraphRAG Answer Log\n\n", encoding="utf-8")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ) as progress:
        overall_task = progress.add_task("[yellow]Overall Benchmark", total=len(config.configs))
        
        artifacts = build_artifacts(config)
        questions = load_benchmark_questions(config)
        results: list[BenchmarkResult] = []
        evaluations: list[EvaluationResult] = []

        for config_id in config.configs:
            progress.update(overall_task, description=f"[cyan]Running {config_id}...")
            
            # Ensure a clean slate for each configuration's database
            if "caracal" in config_id:
                db_path = config.caracal_db_path
                if db_path.is_dir():
                    shutil.rmtree(db_path)
                elif db_path.exists():
                    db_path.unlink()
            
            factory = ADAPTERS[config_id]
            adapter = factory(config)
            try:
                # Use a streaming-enabled version of run_single_config
                result, adapter_evaluations = run_single_config_streaming(
                    adapter,
                    artifacts,
                    questions,
                    config,
                    progress,
                    trace_path,
                    answer_path
                )
                results.append(result)
                evaluations.extend(adapter_evaluations)
                
                # Update partial results report after each config
                (config.output_dir / "comparison_report.txt").write_text(format_report(results), encoding="utf-8")
                write_csv(config.output_dir / "comparison_benchmark.csv", [asdict(r) for r in results])
                
            except MissingExternalService as exc:
                results.append(skipped_result(adapter, artifacts, str(exc)))
            finally:
                adapter.close()
                progress.advance(overall_task)

        # Final summaries
        write_csv(config.output_dir / "evaluation_report.csv", [asdict(e) for e in evaluations])
        generate_quality_summary(config.output_dir / "evaluation_report.csv", config.output_dir / "quality_summary.md")
        (config.output_dir / "database_proposal.md").write_text(format_database_proposal(results), encoding="utf-8")
        
        return results


from dataclasses import dataclass, field

@dataclass
class AggregatedTimings:
    query_planning: list[float] = field(default_factory=list)
    semantic_entry: list[float] = field(default_factory=list)
    query_entity_linking: list[float] = field(default_factory=list)
    semantic_reentry: list[float] = field(default_factory=list)
    relation_expansion: list[float] = field(default_factory=list)
    answer_aware_reranking: list[float] = field(default_factory=list)
    trace_write: list[float] = field(default_factory=list)
    total: list[float] = field(default_factory=list)

def run_single_config_streaming(
    adapter: StorageAdapter,
    artifacts: GraphArtifacts,
    questions: list[BenchmarkQuestion],
    config: BenchmarkConfig,
    progress: Progress,
    trace_path: Path,
    answer_path: Path,
) -> tuple[BenchmarkResult, list[EvaluationResult]]:
    started = time.perf_counter()
    adapter.load(artifacts)
    load_seconds = time.perf_counter() - started

    total_runs = config.repeats * len(questions)
    config_task = progress.add_task(f"  [green]{adapter.config_id}", total=total_runs)

    # Use a lightweight timings aggregator instead of keeping all RetrievalRunResult objects
    timings = AggregatedTimings()
    adapter_evaluations: list[EvaluationResult] = []
    
    total_retrieved_items = 0
    
    # Pre-calculate chunk lookup once per config
    chunk_to_document = {chunk.chunk_id: chunk.document_id for chunk in artifacts.chunks}
    
    with trace_path.open("a", encoding="utf-8") as t_file, \
         answer_path.open("a", encoding="utf-8") as a_file:
         
        for repeat in range(config.repeats):
            for question in questions:
                run = run_retrieval(
                    adapter=adapter,
                    question_id=question.question_id,
                    question=question.question,
                    embedding_dimension=config.embedding_dimension,
                    top_k=config.top_k_semantic_candidates,
                    relation_depth=config.relation_depth,
                    run_id=f"{adapter.config_id}:{question.question_id}:{repeat + 1}",
                    question_type=question.question_type,
                )
                
                # 1. Incremental Answer Generation
                answer, citations = generate_answer_and_citations(run.trace, artifacts)
                adapter.store_answer(answer, citations)
                
                # 2. Stream Trace and Answer
                t_file.write(json.dumps(run.trace.to_json_dict(), ensure_ascii=False) + "\n")
                a_file.write(format_answer_entry(answer, citations))
                
                # 3. Lightweight Evaluation (uses pre-calculated lookup)
                evaluation = evaluate_single_run(run, answer, citations, artifacts, question, chunk_to_document)
                adapter_evaluations.append(evaluation)
                
                # 4. Aggregated lightweight data
                total_retrieved_items += len(run.trace.context_items)
                timings.query_planning.append(run.timing.query_planning_seconds)
                timings.semantic_entry.append(run.timing.semantic_entry_seconds)
                timings.query_entity_linking.append(run.timing.query_entity_linking_seconds)
                timings.semantic_reentry.append(run.timing.semantic_reentry_seconds)
                timings.relation_expansion.append(run.timing.relation_expansion_seconds)
                timings.answer_aware_reranking.append(run.timing.answer_aware_reranking_seconds)
                timings.trace_write.append(run.timing.trace_write_seconds)
                timings.total.append(run.timing.total_seconds)
                
                # Explicitly delete heavy objects to help GC
                del run
                
                progress.advance(config_task)

    progress.remove_task(config_task)

    return (
        BenchmarkResult(
            config_id=adapter.config_id,
            config_name=adapter.config_name,
            graph_db=adapter.graph_db,
            vector_db=adapter.vector_db,
            status="ok",
            total_seconds=load_seconds + sum(timings.total),
            load_seconds=load_seconds,
            semantic_entry_seconds=median(timings.semantic_entry),
            semantic_reentry_seconds=median(timings.semantic_reentry),
            relation_expansion_seconds=median(timings.relation_expansion),
            trace_write_seconds=median(timings.trace_write),
            query_planning_seconds=median(timings.query_planning),
            query_entity_linking_seconds=median(timings.query_entity_linking),
            answer_aware_reranking_seconds=median(timings.answer_aware_reranking),
            documents=len(artifacts.documents),
            chunks=len(artifacts.chunks),
            entities=len(artifacts.entities),
            relationships=len(artifacts.relationships),
            retrieved_context_items=total_retrieved_items,
            citations=sum(e.citations for e in adapter_evaluations),
            adapter_loc=adapter_loc(adapter.__class__),
            semantic_entry_mode=adapter.semantic_entry_mode,
            semantic_reentry_mode=adapter.semantic_reentry_mode,
            relation_expand_mode=adapter.relation_expand_mode,
            cross_store_join_count=adapter.cross_store_join_count,
            sync_notes=adapter.sync_notes,
            notes=adapter.notes,
        ),
        adapter_evaluations,
    )

def evaluate_single_run(run, answer, citations, artifacts, question, chunk_to_document):
    # This just calls existing evaluate_runs but for a single item to keep logic unified
    # but we bypass the bulk table rebuild by passing pre-calculated chunk_to_document if needed
    # For now, let's keep evaluate_runs but wrap it
    return evaluate_runs([run], [(answer, citations)], artifacts, [question])[0]


def format_answer_entry(answer: Answer, citations: list[Citation]) -> str:
    lines = [
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
    if citations:
        for citation in citations:
            lines.append(f"- `{citation.chunk_id}` confidence={citation.confidence:.2f}: {citation.evidence_text}")
    else:
        lines.append("- No citations.")
    lines.append("\n")
    return "\n".join(lines)


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
