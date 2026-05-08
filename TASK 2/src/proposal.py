from __future__ import annotations

from .models import BenchmarkResult


def format_database_proposal(results: list[BenchmarkResult]) -> str:
    caracal = next((result for result in results if result.config_id == "caracal-only"), None)
    external = next((result for result in results if result.config_id == "caracal-external-semantic"), None)
    lines = [
        "# CaracalDB GraphRAG Database Proposal",
        "",
        "## Recommendation",
        "",
        "Use CaracalDB as the authoritative graph ecosystem database for Task 2.",
        "CaracalDB stores source nodes, chunk nodes, entity nodes, embeddings, relation topology, evidence paths, retrieval traces, answers, and citations.",
        "",
        "## Current Benchmark Signal",
        "",
    ]
    if caracal:
        lines.append(
            f"- CaracalDB only: status={caracal.status}, semantic={caracal.semantic_entry_mode}, relation={caracal.relation_expand_mode}, total={caracal.total_seconds}"
        )
    if external:
        lines.append(
            f"- CaracalDB + external semantic index: status={external.status}, semantic={external.semantic_entry_mode}, relation={external.relation_expand_mode}, total={external.total_seconds}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The CaracalDB-only path validates the central proposal when it uses graph-addressable HNSW semantic entry and native relation expansion.",
            "The external semantic index path is useful when semantic candidate generation should be delegated to a specialized index, but its results must re-enter the graph before relation expansion and answer grounding.",
            "",
            "## Remaining External Dependency",
            "",
            "Neo4j comparisons require a running Neo4j server configured through NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD.",
            "",
        ]
    )
    return "\n".join(lines)
