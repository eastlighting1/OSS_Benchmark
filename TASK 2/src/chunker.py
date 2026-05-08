from __future__ import annotations

from .models import Chunk, Document


def chunk_documents(
    documents: list[Document],
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
) -> list[Chunk]:
    if chunk_size_tokens < 1:
        raise ValueError("chunk_size_tokens must be >= 1")
    if chunk_overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")

    chunks: list[Chunk] = []
    step = chunk_size_tokens - chunk_overlap_tokens
    for document in documents:
        tokens = document.text.split()
        if not tokens:
            continue
        for index, start in enumerate(range(0, len(tokens), step)):
            window = tokens[start : start + chunk_size_tokens]
            if not window:
                continue
            chunk_id = f"chunk:{document.document_id.removeprefix('doc:')}:{index:03d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    chunk_index=index,
                    text=" ".join(window),
                    token_count=len(window),
                )
            )
            if start + chunk_size_tokens >= len(tokens):
                break
    return chunks
