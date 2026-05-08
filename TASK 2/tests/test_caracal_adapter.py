from __future__ import annotations

from src.config import BenchmarkConfig
from src.pipeline import build_artifacts
from src.retriever import run_retrieval
from src.storage.caracal_adapter import CaracalStorageAdapter


def test_caracal_adapter_uses_native_graph_ecosystem_primitives(tmp_path) -> None:
    config = BenchmarkConfig(caracal_db_path=tmp_path / "benchmark.crcl")
    artifacts = build_artifacts(config)
    adapter = CaracalStorageAdapter(config.caracal_db_path)

    adapter.load(artifacts)
    result = run_retrieval(
        adapter=adapter,
        question_id="q_test",
        question="How does semantic neighborhood retrieval enter the graph?",
        embedding_dimension=config.embedding_dimension,
        top_k=3,
        relation_depth=2,
        run_id="test-run",
    )

    assert result.trace.semantic_entry_mode in {"caracal_graphrag_search", "caracal_hnsw", "caracal_exact_scan"}
    assert result.trace.relation_expand_mode in {
        "caracal_graphrag_search",
        "caracal_evidence_search",
        "caracal_paths_planned",
        "caracal_neighbors",
        "caracal_neighbors_planned_paths",
        "memory_bfs_fallback",
    }
    assert result.trace.context_items
    assert adapter.capabilities.get("property_index") is True
    assert adapter.native_property_indexes_ready is True
    assert adapter.capabilities.get("text_index") is True
    assert adapter.capabilities.get("graphrag.search") is True
