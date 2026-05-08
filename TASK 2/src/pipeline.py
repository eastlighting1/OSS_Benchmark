from __future__ import annotations

from .chunker import chunk_documents
from .config import BenchmarkConfig
from .document_loader import load_documents
from .datasets.multihoprag import load_multihoprag_documents
from .embeddings import build_embeddings
from .extract_entities import extract_entities
from .extract_relationships import extract_relationships
from .models import GraphArtifacts


def build_artifacts(config: BenchmarkConfig) -> GraphArtifacts:
    if config.dataset == "multihoprag":
        documents = load_multihoprag_documents(
            corpus_path=config.multihoprag_corpus_json,
            questions_path=config.multihoprag_questions_json,
            max_documents=config.max_documents,
            max_questions=config.max_questions,
        )
    else:
        documents = load_documents(config.documents_dir)
    chunks = chunk_documents(documents, config.chunk_size_tokens, config.chunk_overlap_tokens)
    entities, mentions = extract_entities(chunks)
    relationships = extract_relationships(mentions)
    embeddings = build_embeddings(chunks, entities, config.embedding_dimension)
    return GraphArtifacts(
        documents=documents,
        chunks=chunks,
        entities=entities,
        mentions=mentions,
        relationships=relationships,
        embeddings=embeddings,
    )
