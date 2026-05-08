from __future__ import annotations

import hashlib
import math
import os
import re
from functools import lru_cache

from .models import Chunk, EmbeddingRecord, Entity


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def embed_text(text: str, dimension: int) -> list[float]:
    backend = os.getenv("TASK2_EMBEDDING_BACKEND", "hash").strip().lower()
    if backend in {"sentence-transformers", "sentence_transformers", "st"}:
        return _sentence_transformer_embedding(text, dimension)
    return _hash_embedding(text, dimension)


def _hash_embedding(text: str, dimension: int) -> list[float]:
    vector = [0.0] * dimension
    for token in TOKEN_RE.findall(text.lower()):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        raw = int.from_bytes(digest, byteorder="big", signed=False)
        index = raw % dimension
        sign = 1.0 if (raw >> 8) % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _sentence_transformer_embedding(text: str, dimension: int) -> list[float]:
    model = _sentence_transformer_model()
    vector = [float(value) for value in model.encode(text, normalize_embeddings=True).tolist()]
    if len(vector) == dimension:
        return vector
    if len(vector) > dimension:
        vector = vector[:dimension]
    else:
        vector = [*vector, *([0.0] * (dimension - len(vector)))]
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


@lru_cache(maxsize=1)
def _sentence_transformer_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "TASK2_EMBEDDING_BACKEND=sentence-transformers requires the "
            "sentence-transformers package."
        ) from exc
    model_name = os.getenv("TASK2_SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return SentenceTransformer(model_name)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vector dimension mismatch")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def build_embeddings(
    chunks: list[Chunk],
    entities: list[Entity],
    dimension: int,
) -> list[EmbeddingRecord]:
    records = [
        EmbeddingRecord(owner_id=chunk.chunk_id, owner_type="Chunk", vector=embed_text(chunk.text, dimension))
        for chunk in chunks
    ]
    records.extend(
        EmbeddingRecord(
            owner_id=entity.entity_id,
            owner_type="Entity",
            vector=embed_text(f"{entity.name} {entity.description}", dimension),
        )
        for entity in entities
    )
    return records
