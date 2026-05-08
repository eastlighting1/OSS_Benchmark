from __future__ import annotations

import re
from collections import Counter

from .models import Chunk, Entity, EntityMention


DOMAIN_ENTITIES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "GraphRAG": ("concept", "Retrieval augmented generation using graph context.", ("graphrag", "graph rag")),
    "CaracalDB": ("database", "Embedded analytical graph database proposed by this project.", ("caracaldb",)),
    "Neo4j": ("database", "Graph-native comparison database.", ("neo4j",)),
    "VectorDB": ("database", "External semantic index or vector database.", ("vectordb", "vector database", "semantic index")),
    "Chroma": ("database", "Local vector database option.", ("chroma",)),
    "FAISS": ("database", "Lightweight vector search library.", ("faiss",)),
    "Semantic Neighborhood": ("retrieval", "Graph-addressable semantic similarity region.", ("semantic neighborhood", "semantic-neighborhood")),
    "Relation Topology": ("retrieval", "Typed and weighted graph relation structure.", ("relation topology",)),
    "Evidence Path": ("retrieval", "Path connecting retrieved context to supporting source evidence.", ("evidence path", "evidence paths")),
    "Arrow": ("format", "Arrow-native analytical interchange format.", ("arrow",)),
    "HNSW": ("index", "Approximate nearest-neighbor vector index.", ("hnsw",)),
}

COMMON_CAPITALIZED_STARTS = {
    "A",
    "An",
    "And",
    "As",
    "At",
    "But",
    "By",
    "For",
    "From",
    "How",
    "If",
    "In",
    "It",
    "Its",
    "More",
    "New",
    "No",
    "Not",
    "Of",
    "On",
    "Or",
    "The",
    "This",
    "To",
    "What",
    "When",
    "Where",
    "Which",
    "Who",
    "Why",
    "With",
}

CAPITALIZED_PHRASE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?|[A-Z]{2,})"
    r"(?:\s+(?:[A-Z][A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?|[A-Z]{2,})){0,4}\b"
)


def extract_entities(chunks: list[Chunk]) -> tuple[list[Entity], list[EntityMention]]:
    entities_by_id: dict[str, Entity] = {}
    mentions: list[EntityMention] = []
    for chunk in chunks:
        lowered = chunk.text.lower()
        for name, (_entity_type, _description, aliases) in DOMAIN_ENTITIES.items():
            if any(alias in lowered for alias in aliases):
                ensure_entity(entities_by_id, name, DOMAIN_ENTITIES[name][0], DOMAIN_ENTITIES[name][1])
                mentions.append(
                    EntityMention(
                        chunk_id=chunk.chunk_id,
                        entity_id=entity_id(name),
                        mention_text=name,
                    )
                )
        for name in extract_dynamic_names(chunk.text):
            ensure_entity(
                entities_by_id,
                name,
                "named_entity",
                "Capitalized phrase extracted from the source corpus.",
            )
            mentions.append(
                EntityMention(
                    chunk_id=chunk.chunk_id,
                    entity_id=entity_id(name),
                    mention_text=name,
                    confidence=0.75,
                )
            )

    used_entity_ids = {mention.entity_id for mention in mentions}
    return [entity for entity_id, entity in entities_by_id.items() if entity_id in used_entity_ids], mentions


def extract_dynamic_names(text: str, limit: int = 6) -> list[str]:
    counts: Counter[str] = Counter()
    for match in CAPITALIZED_PHRASE.finditer(text):
        name = clean_candidate(match.group(0))
        if not is_useful_candidate(name):
            continue
        counts[name] += 1
    return [name for name, _count in counts.most_common(limit)]


def clean_candidate(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip(" .,:;()[]{}\"'"))


def is_useful_candidate(name: str) -> bool:
    if len(name) < 3:
        return False
    words = name.split()
    if words[0] in COMMON_CAPITALIZED_STARTS and len(words) == 1:
        return False
    if len(words) > 5:
        return False
    if all(word in COMMON_CAPITALIZED_STARTS for word in words):
        return False
    return True


def ensure_entity(
    entities_by_id: dict[str, Entity],
    name: str,
    entity_type: str,
    description: str,
) -> None:
    eid = entity_id(name)
    if eid in entities_by_id:
        return
    entities_by_id[eid] = Entity(
        entity_id=eid,
        name=name,
        canonical_name=canonical_name(name),
        entity_type=entity_type,
        description=description,
    )


def entity_id(name: str) -> str:
    return f"entity:{canonical_name(name)}"


def canonical_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
