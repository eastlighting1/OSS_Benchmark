from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ..models import GraphArtifacts, QueryEntityLink, SemanticCandidate
from .base import merge_query_entity_links
from .caracal_adapter import CaracalStorageAdapter
from .external_semantic_index import ExternalSemanticIndex


class CaracalExternalSemanticAdapter(CaracalStorageAdapter):
    config_id = "caracal-external-semantic"
    config_name = "CaracalDB + external semantic index"
    graph_db = "caracaldb"
    vector_db = "chroma"

    def __init__(self, db_path: Path, vector_store_dir: Path) -> None:
        super().__init__(db_path)
        self.external_index = ExternalSemanticIndex(vector_store_dir / "chroma")
        self._semantic_reentry_bridge: dict[str, list[SemanticCandidate]] = {}

    def load(self, artifacts: GraphArtifacts) -> None:
        super().load(artifacts)
        self.external_index.build(artifacts.embeddings)
        self._semantic_reentry_bridge = build_semantic_reentry_bridge(artifacts)
        self.vector_db = self.external_index.name
        self.sync_notes = f"{self.external_index.name} built from CaracalDB graph node embeddings"

    def semantic_entry(self, question: str, query_embedding: list[float], top_k: int):
        self.semantic_entry_mode = self.external_index.name
        return self.external_index.search(query_embedding, top_k, owner_type="Chunk")

    def semantic_reentry(self, candidates: list[SemanticCandidate]) -> list[SemanticCandidate]:
        self.semantic_reentry_mode = "external_hits_to_caracal_graph_enriched"
        self.cross_store_join_count += len(candidates)
        if not candidates:
            return candidates

        existing_ids = {candidate.node_id for candidate in candidates}
        enriched = list(candidates)

        base_score = candidates[-1].score if candidates else 0.0
        for seed in candidates[: min(4, len(candidates))]:
            for bridged_candidate in self._semantic_reentry_bridge.get(seed.node_id, [])[:3]:
                if bridged_candidate.node_id in existing_ids:
                    continue
                enriched.append(
                    SemanticCandidate(
                        node_id=bridged_candidate.node_id,
                        node_type=bridged_candidate.node_type,
                        score=max(base_score * 0.85, bridged_candidate.score),
                        rank=len(enriched) + 1,
                        reason=f"graph reentry from external semantic hit; {bridged_candidate.reason}",
                        source=bridged_candidate.source,
                    )
                )
                existing_ids.add(bridged_candidate.node_id)

        self.cross_store_join_count += max(0, len(enriched) - len(candidates))
        return enriched

    def link_query_entities(
        self,
        question: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[QueryEntityLink]:
        external_candidates = self.external_index.search(query_embedding, top_k, owner_type="Entity")
        external_links: list[QueryEntityLink] = []
        for index, candidate in enumerate(external_candidates):
            entity = self.entities_by_id.get(candidate.node_id)
            if entity is None:
                continue
            external_links.append(
                QueryEntityLink(
                    entity_id=entity.entity_id,
                    name=entity.name,
                    score=0.35 + candidate.score,
                    rank=index + 1,
                    matched_text="external_entity_embedding",
                    source=f"{self.external_index.name}_entity_reentry",
                )
            )
        native_links = super().link_query_entities(question, query_embedding, top_k)
        self.cross_store_join_count += len(external_links)
        return merge_query_entity_links(external_links, native_links, top_k=top_k)


def build_semantic_reentry_bridge(artifacts: GraphArtifacts) -> dict[str, list[SemanticCandidate]]:
    chunks_by_entity: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for mention in artifacts.mentions:
        chunks_by_entity[mention.entity_id].append((mention.chunk_id, mention.confidence))

    bridged_scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for mention in artifacts.mentions:
        for chunk_id, confidence in chunks_by_entity.get(mention.entity_id, []):
            if chunk_id == mention.chunk_id:
                continue
            bridged_scores[mention.chunk_id][chunk_id] += min(mention.confidence, confidence)

    for relationship in artifacts.relationships:
        source_chunks = chunks_by_entity.get(relationship.source_entity_id, [])
        target_chunks = chunks_by_entity.get(relationship.target_entity_id, [])
        for chunk_id, _ in source_chunks:
            if relationship.evidence_chunk_id != chunk_id:
                bridged_scores[chunk_id][relationship.evidence_chunk_id] += relationship.weight
            for target_chunk_id, _ in target_chunks[:8]:
                if target_chunk_id != chunk_id:
                    bridged_scores[chunk_id][target_chunk_id] += relationship.weight * 0.5

    bridge: dict[str, list[SemanticCandidate]] = {}
    for seed_chunk_id, neighbors in bridged_scores.items():
        ranked = sorted(neighbors.items(), key=lambda item: (-item[1], item[0]))[:8]
        bridge[seed_chunk_id] = [
            SemanticCandidate(
                node_id=chunk_id,
                node_type="Chunk",
                score=0.2 + min(score, 3.0) / 10.0,
                rank=index + 1,
                reason="shared entity or relationship evidence bridge",
                source="caracal_graph_bridge_cache",
            )
            for index, (chunk_id, score) in enumerate(ranked)
        ]
    return bridge
