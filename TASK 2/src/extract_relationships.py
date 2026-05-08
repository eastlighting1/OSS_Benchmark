from __future__ import annotations

from itertools import combinations

from .models import EntityMention, Relationship


def extract_relationships(mentions: list[EntityMention]) -> list[Relationship]:
    by_chunk: dict[str, list[str]] = {}
    for mention in mentions:
        by_chunk.setdefault(mention.chunk_id, [])
        if mention.entity_id not in by_chunk[mention.chunk_id]:
            by_chunk[mention.chunk_id].append(mention.entity_id)

    relationships: dict[str, Relationship] = {}
    for chunk_id, entity_ids in by_chunk.items():
        for left, right in combinations(sorted(entity_ids), 2):
            rel_id = relationship_id(left, right, chunk_id)
            relationships[rel_id] = Relationship(
                relationship_id=rel_id,
                source_entity_id=left,
                target_entity_id=right,
                relationship_type="RELATED_TO",
                evidence_chunk_id=chunk_id,
                weight=1.0,
                description="Entities co-occur in the same chunk.",
            )
    return list(relationships.values())


def relationship_id(left: str, right: str, chunk_id: str) -> str:
    return f"rel:{left.removeprefix('entity:')}:{right.removeprefix('entity:')}:{chunk_id.removeprefix('chunk:')}"
