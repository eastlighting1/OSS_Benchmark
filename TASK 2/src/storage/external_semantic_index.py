from __future__ import annotations

import shutil
from pathlib import Path

from ..models import EmbeddingRecord, SemanticCandidate
from .base import exact_semantic_candidates


class ExternalSemanticIndex:
    """Chroma-backed semantic index with exact in-memory fallback."""

    fallback_name = "local_exact_semantic_index"

    def __init__(self, persist_dir: Path | None = None) -> None:
        self.persist_dir = persist_dir
        self.records: list[EmbeddingRecord] = []
        self.collection = None
        self.name = self.fallback_name

    def build(self, records: list[EmbeddingRecord]) -> None:
        self.records = list(records)
        if self.persist_dir is None:
            return
        try:
            import chromadb
        except Exception:
            return

        if self.persist_dir.exists():
            shutil.rmtree(self.persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.persist_dir))
        collection_name = "task2_semantic_graph_nodes"
        self.collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        for batch in batched(self.records, 2_000):
            self.collection.add(
                ids=[record.owner_id for record in batch],
                embeddings=[record.vector for record in batch],
                metadatas=[
                    {
                        "owner_type": record.owner_type,
                        "model_name": record.model_name,
                    }
                    for record in batch
                ],
                documents=[record.owner_id for record in batch],
            )
        self.name = "chroma_persistent"

    def search(self, query_embedding: list[float], top_k: int, owner_type: str | None = None) -> list[SemanticCandidate]:
        if self.collection is not None:
            where = None if owner_type is None else {"owner_type": owner_type}
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["distances", "metadatas"],
                where=where,
            )
            ids = result.get("ids", [[]])[0]
            distances = result.get("distances", [[]])[0]
            metadatas = result.get("metadatas", [[]])[0]
            candidates: list[SemanticCandidate] = []
            for index, node_id in enumerate(ids):
                distance = float(distances[index])
                metadata = metadatas[index] or {}
                candidates.append(
                    SemanticCandidate(
                        node_id=node_id,
                        node_type=str(metadata.get("owner_type", node_id.split(":", 1)[0].title())),
                        score=1.0 - distance,
                        rank=index + 1,
                        reason="Chroma persistent semantic index candidate",
                        source=self.name,
                    )
                )
            return candidates

        return exact_semantic_candidates(
            records=[record for record in self.records if owner_type is None or record.owner_type == owner_type],
            query_embedding=query_embedding,
            top_k=top_k,
            source=self.fallback_name,
            reason="external semantic index exact scan",
        )


def batched(records: list[EmbeddingRecord], size: int) -> list[list[EmbeddingRecord]]:
    return [records[index : index + size] for index in range(0, len(records), size)]
