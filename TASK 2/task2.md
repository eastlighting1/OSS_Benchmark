# caracalDB-Based GraphRAG Implementation System

## 1. Implementation Specification

### 1.1 Project Overview

This project aims to implement a toy GraphRAG system that combines document retrieval, knowledge graph construction, graph-aware context expansion, and answer generation.

The project proposes `caracaldb` as the main graph-oriented storage and query database. The system should demonstrate how documents can be transformed into chunks, entities, semantic neighborhoods, evidence paths, and relationships that can be retrieved as one graph ecosystem.

The purpose of this project is not to build a production-grade enterprise RAG platform, but to demonstrate GraphRAG data modeling, document ETL, graph construction, retrieval design, answer grounding, and database selection reasoning.

### 1.2 Core Concept

GraphRAG is a retrieval-augmented generation approach that uses a knowledge graph to improve context retrieval.

Instead of retrieving only semantically similar text chunks, the system should also retrieve related entities, neighboring facts, communities, source documents, and graph paths. This allows the answer generation layer to use richer context than a flat vector search pipeline.

The system should support the following flow:

```text
Raw Documents
   ->
Document Chunking
   ->
Entity / Relationship Extraction
   ->
Embedding Generation
   ->
caracaldb Graph Storage
   ->
Hybrid Retrieval
   ->
Graph Context Expansion
   ->
Grounded Answer Generation
   ->
Evaluation / Comparison Output
```

### 1.3 Dataset

The recommended dataset is:

```text
Small document corpus for GraphRAG demonstration
```

The corpus may be one of the following:

```text
Project documents
Company policy documents
Research paper abstracts
Wikipedia-style topic documents
Technical documentation pages
News article snippets
```

The system assumes the following input structure:

```text
data/raw/documents/
|-- document_001.md
|-- document_002.md
|-- document_003.txt
`-- metadata.csv
```

Each document should include or be associated with:

```text
document_id
title
source_path
author or source
created_at or published_at
document text
optional tags
```

### 1.4 Technology Stack

The system should use the following technologies:

```text
Programming Language: Python
Proposed Database: caracaldb
Retrieval Pattern: GraphRAG
Document Format: Markdown, TXT, or CSV
Output Format: CLI output, JSON reports, CSV reports, and Markdown answer logs
```

`caracaldb` is the proposed database for this project. It should be used as the main storage layer for graph entities, document chunks, relationships, and retrieval metadata.

Optional components may include:

```text
Embedding Model: sentence-transformers, OpenAI embeddings, or local mock embeddings
LLM Provider: OpenAI API, local LLM, or deterministic mock generator
Evaluation Library: custom Python evaluation scripts
```

The implementation must be able to run in a local development environment. If external API keys are not configured, the system should provide a mock or rule-based fallback mode.

### 1.5 System Architecture

The system should follow this pipeline:

```text
1. Load raw documents
2. Clean and normalize text
3. Split documents into chunks
4. Extract entities from chunks
5. Extract relationships between entities
6. Generate or simulate embeddings
7. Store documents, chunks, entities, relationships, and embeddings in caracaldb
8. Execute hybrid GraphRAG retrieval
9. Expand context using graph neighbors and paths
10. Generate grounded answers with citations
11. Evaluate retrieval and answer quality
12. Produce a database proposal and comparison report
```

Overall flow:

```text
Document Corpus
   ->
Text Processing Layer
   ->
Entity / Relationship Extraction
   ->
Embedding Layer
   ->
caracaldb
   ->
Hybrid Retriever
   ->
Graph Context Builder
   ->
Answer Generator
   ->
Evaluation Reports
```

### 1.6 Data Model

The system must include at least the following entities:

```text
Document
Chunk
Entity
Relationship
Question
Answer
Citation
```

Entity meanings:

```text
Document: Original source document
Chunk: Text segment derived from a document
Entity: Named concept, organization, person, technology, place, or domain term
Relationship: Directed or undirected connection between two entities
Question: User query submitted to the GraphRAG system
Answer: Generated answer from retrieved context
Citation: Evidence linking answer statements to source chunks
```

### 1.7 Entity Schema

Document:

```text
document_id: string
title: string
source_path: string
source_type: string
created_at: datetime
metadata_json: string
```

Chunk:

```text
chunk_id: string
document_id: string
chunk_index: integer
text: string
token_count: integer
embedding_id: string
```

Entity:

```text
entity_id: string
name: string
entity_type: string
description: string
canonical_name: string
```

Relationship:

```text
relationship_id: string
source_entity_id: string
target_entity_id: string
relationship_type: string
weight: float
evidence_chunk_id: string
description: string
```

Question:

```text
question_id: string
question_text: string
created_at: datetime
retrieval_mode: string
```

Answer:

```text
answer_id: string
question_id: string
answer_text: string
grounding_score: float
created_at: datetime
```

Citation:

```text
citation_id: string
answer_id: string
chunk_id: string
evidence_text: string
confidence: float
```

### 1.8 Relationship Model

The system must support at least the following relationships:

```text
Document - HAS_CHUNK - Chunk
Chunk - MENTIONS - Entity
Entity - RELATED_TO - Entity
Entity - SUPPORTS_ANSWER - Answer
Answer - CITES - Chunk
Question - PRODUCES - Answer
```

Relationship meanings:

```text
HAS_CHUNK: A document contains a chunk
MENTIONS: A chunk mentions an extracted entity
RELATED_TO: Two entities have a discovered semantic or factual relationship
SUPPORTS_ANSWER: An entity contributes to an answer
CITES: An answer cites a source chunk
PRODUCES: A question produces an answer
```

### 1.9 caracaldb Storage Structure

The system should store graph nodes and edges in `caracaldb`.

Required storage objects:

```text
documents
chunks
entities
relationships
chunk_entities
questions
answers
citations
embeddings
retrieval_runs
retrieval_results
```

`documents`:

```text
document_id
title
source_path
source_type
created_at
metadata_json
```

`chunks`:

```text
chunk_id
document_id
chunk_index
text
token_count
embedding_id
```

`entities`:

```text
entity_id
name
canonical_name
entity_type
description
```

`relationships`:

```text
relationship_id
source_entity_id
target_entity_id
relationship_type
weight
evidence_chunk_id
description
```

`chunk_entities`:

```text
chunk_id
entity_id
mention_text
start_offset
end_offset
confidence
```

`embeddings`:

```text
embedding_id
owner_type
owner_id
model_name
vector_json
dimension
```

`retrieval_runs`:

```text
run_id
question_id
retrieval_mode
created_at
top_k_semantic_candidates
relation_depth
```

`retrieval_results`:

```text
run_id
result_id
result_type
result_id_ref
score
rank
reason
```

### 1.10 Document Processing Specification

The document processing layer must perform the following tasks:

```text
Load Markdown, TXT, and CSV-based documents
Normalize whitespace and remove unusable content
Split long documents into overlapping chunks
Create stable document and chunk IDs
Extract candidate entities
Normalize duplicate entity names
Extract relationships between entities
Generate or simulate embeddings
Validate that all relationships reference existing entities and chunks
```

Processing rules:

```text
Chunk size must be configurable.
Chunk overlap must be configurable.
Every chunk must belong to exactly one document.
Every entity mention must point to a valid chunk.
Every relationship must point to valid source and target entities.
The system must preserve source traceability from answer to chunk to document.
The extraction layer may use an LLM, rule-based extractor, or deterministic test extractor.
The default local mode should work without paid external APIs.
```

Recommended defaults:

```text
chunk_size_tokens: 500
chunk_overlap_tokens: 80
top_k_semantic_candidates: 8
relation_depth: 2
max_relation_neighbors: 20
embedding_dimension: 384 or mock equivalent
```

### 1.11 Main Features

#### Feature 1: Document Ingestion

The system must read a document corpus, split documents into chunks, and store the processed results in `caracaldb`.

Input:

```text
data/raw/documents/
```

Output:

```text
documents
chunks
embeddings
```

#### Feature 2: Entity and Relationship Extraction

The system must extract entities and relationships from document chunks.

Expected output:

```text
entity_id
name
entity_type
relationship_type
source_entity
target_entity
evidence_chunk
```

#### Feature 3: Graph Ecosystem Retrieval

The system must support at least three retrieval modes:

```text
semantic_neighborhood
relation_topology
graph_ecosystem
```

Retrieval behavior:

```text
semantic_neighborhood: enter the graph through embedding similarity, semantic buckets, or SEMANTIC_NEIGHBOR edges
relation_topology: retrieve context through entity mentions, typed relations, evidence paths, and provenance edges
graph_ecosystem: combine semantic entry, relation topology, evidence paths, and citation-aware scoring
```

Expected output:

```text
question
retrieval_mode
ranked_chunks
matched_entities
expanded_entities
relation_paths
evidence_paths
scores
```

The retriever must include the following GraphRAG-specific stages:

```text
query_entity_linking: link query spans to graph Entity nodes using lexical and embedding signals
question_type_strategy: choose retrieval budgets and path strategy from inference/comparison/temporal/null types
multi_hop_evidence_path_planning: expand from semantic Chunk hits and linked Entity seeds into evidence Chunk paths
answer_aware_reranking: rerank context using question overlap, linked entities, predicted answer hints, and path score
evidence_grounded_answer_span_extraction: extract final answer candidates only from retrieved evidence chunks
citation_reranking: choose citation chunks from reranked evidence, with optional source diversity
answer_correctness_evaluation: compare predicted answer against dataset gold answer using EM, contains, and token F1
```

These stages are adapter and benchmark responsibilities. CaracalDB should support them through generic entity lookup, vector search, and path traversal primitives, not by embedding GraphRAG-specific policies in the database core.

#### Feature 4: Relation and Evidence Expansion

Given initial semantic candidates or matched entities, the system must expand context through relation topology and evidence paths.

Expansion factors:

```text
Mentioned entities
Direct entity relationships
Semantic-neighborhood edges
Neighbor chunks mentioning related entities
Relationship evidence chunks
Citation and provenance paths
Optional community summaries
```

Default scoring formula:

```text
context_score =
    2.0 * semantic_neighborhood_score
  + 1.5 * entity_match_score
  + 1.0 * relation_path_score
  + 0.8 * evidence_path_score
  + 0.5 * source_diversity_score
```

Expected output:

```text
context_item_id
context_type
source_chunk_id
score
reason
source_document
```

#### Feature 5: Grounded Answer Generation

The system must generate answers using retrieved and expanded context.

The answer generator must:

```text
Use only retrieved context when possible
Return citations for answer evidence
Clearly state when context is insufficient
Avoid unsupported claims
Persist question, answer, retrieval run, and citations
```

Expected output:

```text
answer_text
citations
used_chunks
used_entities
grounding_score
```

#### Feature 6: GraphRAG Evaluation

The system must evaluate retrieval and answer quality.

Metrics should include at least:

```text
retrieval_precision_at_k
context_recall
citation_coverage
answer_grounding_score
latency_seconds
evidence_recall_at_context
citation_recall
answer_exact_match
answer_contains_gold
answer_token_f1
```

Expected output:

```text
evaluation_report.csv
retrieval_trace.json
answer_log.md
```

### 1.12 Proposal and Comparison Configurations

The project should compare four GraphRAG configurations.

The first two configurations are not weak baselines and should not be described as simply replacing an RDB or a VectorDB. They are `Graph DB only` systems where persistence, semantic retrieval, relation traversal, evidence, and provenance are all modeled as parts of one graph ecosystem.

```text
Config 1: CaracalDB only
Config 2: Neo4j only
Config 3: CaracalDB + VectorDB
Config 4: Neo4j + VectorDB
```

Configuration meanings:

```text
Config 1. CaracalDB only
  Use CaracalDB as the full GraphRAG graph ecosystem:
  source nodes, chunk nodes, entity nodes, relation topology,
  embedding-bearing nodes, semantic-neighborhood indexes or edges,
  evidence paths, answers, citations, and retrieval traces.

Config 2. Neo4j only
  Use Neo4j as the full GraphRAG graph ecosystem:
  document/chunk/entity nodes, typed relationships,
  embedding properties or vector indexes, semantic neighborhoods,
  relation path expansion, answers, citations, and traces.

Config 3. CaracalDB + VectorDB
  Use CaracalDB as the authoritative graph ecosystem.
  Use a VectorDB as an external semantic index that returns graph node IDs.
  Re-enter external semantic hits into CaracalDB as semantic candidate relations and retrieval traces.

Config 4. Neo4j + VectorDB
  Use Neo4j as the authoritative graph ecosystem.
  Use a VectorDB as an external semantic index that returns graph node IDs.
  Re-enter external semantic hits into Neo4j as semantic candidate relations and retrieval traces.
```

Recommended selected components:

```text
Main proposed DB: CaracalDB
Comparison Graph DB: Neo4j
External semantic index option: Chroma by default, FAISS as lightweight fallback
```

### 1.13 Core vs Adapter Responsibility Boundary

Because CaracalDB is also maintained by the project owner, the implementation must distinguish between database-core features and GraphRAG application adapter features.

General rule:

```text
CaracalDB core:
  General graph, vector, analytical, and storage primitives that are useful beyond GraphRAG.

A-stage adapter:
  GraphRAG-specific schema mapping, retrieval policy, scoring, fallback logic, and orchestration.
```

Boundary table:

| Capability | CaracalDB Core | A-Stage Adapter | Notes |
| ---------- | -------------- | --------------- | ----- |
| Node/edge table storage | Required | Schema mapping | Core stores data; adapter decides GraphRAG classes and edge types. |
| Arrow-native table import/export | Required | Batch preparation | Core should expose stable Arrow APIs. |
| Document loading | Not core | Required | File parsing and corpus policy are application logic. |
| Chunking | Not core | Required | Chunk size, overlap, and text normalization are GraphRAG policy. |
| Entity extraction | Not core | Required | LLM/rule/model extraction is application logic. |
| Relationship extraction | Not core | Required | Relationship semantics depend on the corpus and task. |
| Embedding generation | Not core | Required | Model choice belongs to the application layer. |
| Embedding storage | Required | Formatting | Core should store vector-like properties or embedding nodes. |
| Vector index/search | Core candidate | Fallback allowed | HNSW should be exposed as a graph-addressable semantic-neighborhood primitive if suitable. |
| Exact cosine brute force | Not core | Fallback | Temporary adapter fallback for small corpora. |
| Similarity edges | Edge storage core | Edge creation adapter | Core stores edges; adapter decides which chunks/entities become SEMANTIC_NEIGHBORs. |
| Relation topology indexes | Core candidate | Relation policy | Core accelerates typed relation traversal; adapter decides relation semantics. |
| Evidence/provenance paths | Storage and traversal core | Trace schema | Core supports path storage/query; adapter assigns GraphRAG evidence meaning. |
| k-hop/neighbors traversal | Core candidate | BFS fallback | Generic graph primitive, not GraphRAG-only. |
| shortest_path | Core candidate | Optional fallback | Useful for graph workloads beyond GraphRAG. |
| GraphRAG scoring formula | Not core | Required | Semantic, entity, path, and citation weights are application policy. |
| Retrieval trace persistence | Storage core | Trace schema | Core stores records; adapter defines trace meaning. |
| Answer generation | Not core | Required | LLM or deterministic answer generation is application logic. |
| Citation generation | Not core | Required | Citation quality and answer grounding are application logic. |
| Explain/profile | Core candidate | Benchmark use | Needed to explain optimization and benchmark behavior. |

CaracalDB should not directly implement a `GraphRAG feature`. It should expose graph, vector, and analytical primitives that make GraphRAG easy to build.

### 1.14 Graph Ecosystem Optimization Strategy

In Config 1 and Config 2, the graph database is not a container that imitates separate RDB and VectorDB products. It should model GraphRAG as a graph ecosystem where semantic similarity, explicit relations, source evidence, and answer provenance are connected and queryable together.

```text
Graph-native persistence:
  Documents, chunks, questions, answers, citations, retrieval runs,
  and evaluation records are first-class graph objects.

Graph-native semantic retrieval:
  Embeddings, vector indexes, semantic buckets, and similarity edges
  form graph-addressable semantic neighborhoods.

Graph-native relation retrieval:
  Entity relations, chunk mentions, source evidence, and citation paths
  form typed, weighted, traversable relation topology.
```

For CaracalDB only, the preferred strategy is:

```text
1. Store documents, chunks, entities, answers, citations, and retrieval runs as graph nodes.
2. Store HAS_CHUNK, MENTIONS, RELATED_TO, PRODUCES, CITES, EVIDENCED_BY, and SEMANTIC_NEIGHBOR as graph edges.
3. Store embeddings as graph-addressable properties or Embedding nodes linked to Chunk and Entity nodes.
4. Use public HNSW search as a semantic-neighborhood entry point when the API is available.
5. Convert vector hits into candidate graph objects with score, rank, and reason metadata.
6. Materialize high-confidence semantic neighborhoods as SEMANTIC_NEIGHBOR edges when useful.
7. Use Tuft patterns to combine semantic candidates with entity relations and evidence paths.
8. Use application-level BFS only as a temporary traversal fallback for missing variable-length path APIs.
```

For Neo4j only, the preferred strategy is:

```text
1. Store documents, chunks, entities, answers, citations, and retrieval runs as graph nodes.
2. Store GraphRAG relations as typed Neo4j relationships with evidence and weight properties.
3. Store embeddings as node properties and index them as semantic entry points.
4. Convert vector index hits into candidate graph paths.
5. Use Cypher variable-length paths to connect semantic candidates, entities, evidence chunks, and citations.
6. Use indexes on document_id, chunk_id, entity_id, canonical_name, and retrieval run IDs.
```

The important comparison principle is:

```text
Graph DB only systems should be optimized as graph ecosystems.
They should expose semantic similarity as graph neighborhoods and relations as evidence paths,
not as isolated table joins or detached vector top-k calls.
```

### 1.15 Recommended Module Structure

```text
graphrag_caracal_project/
|-- data/
|   |-- raw/
|   |   |-- documents/
|   |   |   |-- document_001.md
|   |   |   |-- document_002.md
|   |   |   `-- metadata.csv
|   |   `-- questions.csv
|   `-- processed/
|       |-- documents.csv
|       |-- chunks.csv
|       |-- entities.csv
|       |-- relationships.csv
|       |-- chunk_entities.csv
|       |-- embeddings.csv
|       |-- graphrag.caracal/
|       |-- neo4j_export/
|       `-- vector_store/
|-- src/
|   |-- config.py
|   |-- document_loader.py
|   |-- chunker.py
|   |-- extract_entities.py
|   |-- extract_relationships.py
|   |-- embeddings.py
|   |-- schema.py
|   |-- storage/
|   |   |-- base.py
|   |   |-- caracal_adapter.py
|   |   |-- neo4j_adapter.py
|   |   |-- external_semantic_index.py
|   |   |-- caracal_external_semantic_adapter.py
|   |   `-- neo4j_external_semantic_adapter.py
|   |-- retriever.py
|   |-- relation_expander.py
|   |-- context_builder.py
|   |-- answer_generator.py
|   |-- evaluator.py
|   |-- compare_configs.py
|   `-- pipeline.py
|-- outputs/
|   |-- retrieval_trace.json
|   |-- answer_log.md
|   |-- evaluation_report.csv
|   |-- database_proposal.md
|   |-- comparison_report.txt
|   `-- comparison_benchmark.csv
|-- tests/
|   |-- test_chunker.py
|   |-- test_entity_extraction.py
|   |-- test_relationship_extraction.py
|   |-- test_retriever.py
|   |-- test_storage_adapters.py
|   `-- test_answer_grounding.py
|-- README.md
`-- main.py
```

### 1.16 Storage Adapter Interface

All comparison configurations should implement the same storage adapter interface.

```text
StorageAdapter
|-- store_documents(docs)
|-- store_chunks(chunks)
|-- store_entities(entities)
|-- store_relationships(relationships)
|-- store_embeddings(embeddings)
|-- semantic_entry(question, query_embedding, top_k)
|-- semantic_reentry(candidates)
|-- relation_expand(seed_node_ids, depth)
|-- build_evidence_paths(context_items)
|-- store_retrieval_trace(trace)
|-- store_answer(answer, citations)
`-- get_retrieval_trace(run_id)
```

Adapter mapping:

```text
CaracalStorageAdapter:
  Full graph-ecosystem CaracalDB implementation.
  semantic_entry uses graph-addressable HNSW, semantic buckets, or temporary Arrow/numpy candidate generation.
  relation_expand combines semantic candidates, entity topology, evidence edges, and citation paths through Tuft or traversal APIs.
  fallback code must re-enter candidates as graph-scored context items, not bypass the graph.

Neo4jStorageAdapter:
  Full graph-ecosystem Neo4j implementation.
  semantic_entry uses Neo4j vector indexes as graph entry points.
  relation_expand uses Cypher paths to connect semantic candidates, entities, evidence chunks, and citations.

CaracalExternalSemanticAdapter:
  Extends CaracalStorageAdapter.
  Uses Chroma or FAISS as an external semantic index.
  External semantic index results must be mapped back to CaracalDB Chunk or Entity nodes.
  The adapter persists SEMANTIC_CANDIDATE or retrieval-result edges so semantic retrieval remains part of the graph ecosystem.

Neo4jExternalSemanticAdapter:
  Extends Neo4jStorageAdapter.
  Uses Chroma or FAISS as an external semantic index.
  External semantic index results must be mapped back to Neo4j Chunk or Entity nodes.
  The adapter persists semantic candidate relations or retrieval traces so external semantic hits participate in graph traversal.
```

### 1.17 Execution

The full pipeline should be executable with:

```bash
python main.py --documents data/raw/documents
```

Optional commands may include:

```bash
python main.py ingest --documents data/raw/documents
python main.py build-graph
python main.py retrieve --question "What does the corpus say about GraphRAG?"
python main.py ask --question "Why is caracaldb suitable for this project?"
python main.py ask --question "Which entities are connected to retrieval quality?"
python main.py trace --question-id q_001
python main.py evaluate
python main.py compare-db
python main.py compare-configs
python main.py compare-configs --configs caracal-only,neo4j-only,caracal-vector,neo4j-vector
```

### 1.18 Deliverables

The final submission must include:

```text
Source code
README with setup and execution instructions
caracaldb data files or database generation scripts
Document ingestion and chunking code
Entity and relationship extraction code
Hybrid GraphRAG retrieval implementation
Answer generation with citations
Sample questions and answers
Retrieval traces
Evaluation report
Database proposal report explaining caracaldb
Core vs adapter responsibility document
CaracalDB core development requirements document
Four-configuration comparison benchmark
Basic tests
```

# 2. Requirements

## 2.1 Functional Requirements

### FR-001. Document Loading

The system shall load documents from a local folder.

Acceptance criteria:

```text
The system can read Markdown and TXT files.
The system can optionally read CSV metadata.
File paths are provided through configuration or CLI arguments.
If loading fails, the system returns a clear error message.
```

### FR-002. Text Cleaning and Chunking

The system shall clean document text and split it into retrievable chunks.

Acceptance criteria:

```text
Each document produces one or more chunks.
Chunk IDs are stable across repeated runs.
Chunk size and overlap are configurable.
Empty chunks are removed.
```

### FR-003. Entity Extraction

The system shall extract entities from document chunks.

Acceptance criteria:

```text
The entities table stores one row per canonical entity.
Entity mentions are linked to chunks.
Duplicate entity names are normalized.
Entity extraction can run in deterministic local mode.
```

### FR-004. Relationship Extraction

The system shall extract relationships between entities.

Acceptance criteria:

```text
Relationships include source entity, target entity, relationship type, and evidence chunk.
Relationships do not reference missing entities.
Relationship extraction can run in deterministic local mode.
Repeated pipeline execution does not create unintended duplicate relationships.
```

### FR-005. caracaldb Storage

The system shall store documents, chunks, entities, relationships, embeddings, retrieval runs, answers, and citations in `caracaldb`.

Acceptance criteria:

```text
All required storage objects are created.
The number of stored rows matches the processed data.
The database can be rebuilt from raw documents.
The database supports retrieval queries needed by the GraphRAG pipeline.
```

### FR-006. Semantic Neighborhood Retrieval

The system shall create graph-addressable semantic entry candidates using embeddings, semantic buckets, vector indexes, or deterministic mock similarity.

Acceptance criteria:

```text
The user can submit a natural-language question.
The retriever returns ranked Chunk or Entity node candidates.
The result includes score, rank, source document, and semantic-entry reason.
Semantic candidates can be re-entered into the graph as retrieval-result or SEMANTIC_CANDIDATE relations.
The system works even when external embedding APIs are unavailable.
```

### FR-007. Relation Topology Retrieval

The system shall retrieve context through typed relation topology, evidence paths, and provenance edges.

Acceptance criteria:

```text
The retriever identifies entities mentioned in the question.
The retriever finds related entities, evidence chunks, and citation paths in the graph.
The retriever returns chunks connected to matched or expanded entities through explainable paths.
The relation depth can be configured.
```

### FR-008. Hybrid GraphRAG Retrieval

The system shall combine semantic neighborhoods, relation topology, evidence paths, and citation-aware scoring.

Acceptance criteria:

```text
The system supports semantic_neighborhood, relation_topology, and graph_ecosystem modes.
Hybrid results include semantic-entry reasons, relation-path reasons, and evidence-path reasons.
Duplicate context items are merged.
Results are sorted by final context score.
```

### FR-009. Answer Generation With Citations

The system shall generate grounded answers from retrieved context.

Acceptance criteria:

```text
The answer includes citations to source chunks.
The answer states when the retrieved context is insufficient.
The answer record is persisted in the database.
Each citation points to a valid chunk.
```

### FR-010. Retrieval Trace

The system shall produce a trace explaining how context was retrieved.

Acceptance criteria:

```text
The trace includes initial chunks, matched entities, expanded entities, graph paths, and final context.
The trace can be exported as JSON.
The trace can be inspected for debugging and evaluation.
```

### FR-011. Evaluation

The system shall evaluate GraphRAG quality.

Acceptance criteria:

```text
The system can run a fixed evaluation question set.
The evaluation reports retrieval precision, citation coverage, grounding score, and latency.
The evaluation output is exported as CSV.
```

### FR-012. Database Proposal and Comparison

The system shall provide a database proposal explaining why `caracaldb` is selected.

Acceptance criteria:

```text
The proposal compares four configurations: CaracalDB only, Neo4j only, CaracalDB + VectorDB, and Neo4j + VectorDB.
The comparison treats Graph DB only systems as optimized single-database GraphRAG architectures.
The comparison addresses graph-native persistence, semantic neighborhoods, relation topology, evidence paths, provenance, setup complexity, and extensibility.
The report clearly states why caracaldb is proposed for this GraphRAG project.
The comparison includes a runnable or semi-runnable benchmark where possible.
```

### FR-013. CaracalDB Core and Adapter Boundary

The system shall document which features belong in CaracalDB core and which features belong in the GraphRAG adapter layer.

Acceptance criteria:

```text
GraphRAG-specific logic such as chunking, extraction, scoring, answer generation, and citation policy stays in the adapter layer.
General graph/vector/storage primitives such as vector index/search, k-hop traversal, neighbors, batch upsert, and explain/profile are identified as CaracalDB core candidates.
Temporary fallbacks such as numpy vector search and Python BFS are labeled as adapter fallbacks.
The benchmark report distinguishes native CaracalDB behavior from adapter fallback behavior.
```

## 2.2 Non-Functional Requirements

### NFR-001. Reproducibility

Given the same input documents and configuration, the system shall produce the same chunks, entities, relationships, retrieval traces, and evaluation results in deterministic mode.

### NFR-002. Modularity

Document loading, chunking, extraction, storage, semantic entry, relation expansion, evidence path construction, answer generation, and evaluation logic shall be separated into independent modules.

### NFR-003. Extensibility

The system shall be designed so the following capabilities can be added later:

```text
Community detection
Graph summarization
Incremental document updates
Multi-hop reasoning
Human feedback on citations
Real vector database adapter
Graph visualization
Native CaracalDB vector index API
Native CaracalDB k-hop or neighbors API
```

### NFR-004. Maintainability

Functions and classes shall use clear names. Core logic shall include comments or docstrings where helpful.

### NFR-005. Error Handling

The system shall handle the following cases:

```text
Missing input folder
Unsupported document format
Empty document
Chunking failure
Entity extraction failure
Embedding generation failure
Database connection failure
Empty retrieval result
Insufficient context for answer generation
```

### NFR-006. Performance

For a small local document corpus, the full pipeline should run comfortably in a typical local development environment.

Recommended targets:

```text
Document loading and chunking: within 1 minute
Entity and relationship extraction in deterministic mode: within 2 minutes
Database loading: within 1 minute
Single retrieval query: within 5 seconds
Answer generation with mock mode: within 5 seconds
```

### NFR-007. Testability

Core transformation and retrieval functions shall be unit-testable.

Test targets include:

```text
Document loading
Chunk boundary generation
Stable ID creation
Entity normalization
Relationship validation
Graph neighbor expansion
Hybrid score calculation
Citation validation
Insufficient-context handling
```

# 3. caracalDB Proposal

## 3.1 Purpose

This project proposes `caracaldb` as the primary database for a local GraphRAG implementation.

The database should store both document-oriented data and graph-oriented data:

```text
Documents
Chunks
Entities
Relationships
Embeddings
Retrieval traces
Answers
Citations
```

The goal is to show that a GraphRAG system does not only need vector search. It also needs persistent, inspectable, relationship-aware storage that can connect source text, extracted entities, graph edges, and generated answers.

## 3.2 Why caracaldb Fits This Project

`caracaldb` is proposed because it can serve as the project database for structured GraphRAG artifacts.

Advantages:

```text
It can store document chunks, entities, and relationships in one local project database.
It supports inspectable graph artifacts needed for reproducible ETL and retrieval outputs.
It can represent knowledge graph edges through relationship tables.
It keeps retrieval traces and citations inspectable.
It is suitable for a toy project where local execution and clear artifacts matter.
It creates a distinctive database choice compared with common SQLite-only RAG demos.
```

Limitations:

```text
Variable-length paths such as *1..N may require application-side logic until native APIs are available.
If HNSW is implemented internally but not exposed as a public query API, vector similarity may need an adapter fallback.
The project must clearly document which retrieval responsibilities belong to caracaldb and which belong to the application layer.
```

Recommended use in this project:

```text
Persistent storage for GraphRAG artifacts
Entity and relationship lookup
Chunk-to-entity and entity-to-chunk retrieval
Retrieval trace storage
Citation traceability
Database proposal and comparison target
```

## 3.3 CaracalDB Capability Assumptions

The proposal assumes the following CaracalDB capabilities are available or planned in the current development context:

```text
Node/edge table storage
Arrow-native table import/export
Tuft query support for graph-style patterns
Multi-hop pattern matching for explicit patterns
degree() graph built-in
CSR/CSC graph indexes for traversal-oriented execution
HNSW implementation in the graph module
Expand and Join physical operators
```

The implementation must verify which of these capabilities are exposed through stable public APIs:

```text
Public vector index creation API
Public vector search API
Tuft-level vector search or vector distance functions
Python-level access to CSR/CSC neighbor expansion
Native k-hop or neighbors API
Native shortest_path API
Explain/profile API for benchmark interpretation
```

## 3.4 CaracalDB Core Roadmap Candidates

The following features should be considered CaracalDB core candidates because they are general database primitives, not GraphRAG-only application logic:

```text
Public HNSW vector index API
Vector property type or vector-compatible column convention
cosine, dot product, and L2 distance functions
Vector search as a graph-addressable semantic entry point over node classes or vector columns
Semantic-neighborhood edge materialization support
k-hop traversal API
neighbors API
shortest_path API
Batch upsert for node and edge tables
Property and type indexes for IDs and canonical names
Explain/profile output for Tuft and physical execution plans
```

These features would make GraphRAG easier, but they should remain general enough for other graph analytics, recommendation, search, and entity-resolution workloads.

## 3.5 Adapter-Level Fallbacks

The A-stage GraphRAG adapter may implement fallbacks when CaracalDB core APIs are not yet exposed.

Allowed fallbacks:

```text
Arrow/numpy brute-force vector search for small corpora
Python BFS for variable-depth relation expansion
Precomputed SEMANTIC_NEIGHBOR edges for semantic neighborhoods
Application-side score merging for semantic, entity, path, and citation signals
Application-side retrieval trace construction
```

Fallbacks must be labeled clearly in benchmark outputs.

Example labels:

```text
caracal_vector_mode=native_hnsw
caracal_vector_mode=arrow_numpy_fallback
caracal_relation_expand_mode=tuft_pattern
caracal_relation_expand_mode=python_bfs_fallback
```

The distinction matters because a fallback may later become a CaracalDB core feature.

## 3.6 Four Configuration Comparison Table

| Criterion | Config 1 CaracalDB Only | Config 2 Neo4j Only | Config 3 CaracalDB + VectorDB | Config 4 Neo4j + VectorDB |
| --------- | ----------------------: | ------------------: | -----------------------------: | -------------------------: |
| Proposed main architecture | Very High | Low | High | Medium |
| Single-database full process | Very High | Very High | Low | Low |
| Embedded/local convenience | Very High | Low-Medium | High | Low-Medium |
| Graph artifact persistence | High | Medium-High | High | Medium-High |
| Relation topology traversal | Medium-High | Very High | Medium-High | Very High |
| Semantic neighborhood retrieval | Verify API | High | Very High via external semantic index | Very High via external semantic index |
| Provenance and trace storage | High | High | High | High |
| Operational complexity | Low | Medium | Medium | High |
| Benchmark fairness | High | High | High | High |
| Portfolio distinctiveness | Very High | High | Very High | High |

## 3.7 Database and Configuration Evaluation

### caracaldb

`caracaldb` is the proposed database for this GraphRAG project.

Advantages:

```text
It stores the complete GraphRAG state in one project database.
It supports graph-connected documents, chunks, entities, relationships, answers, and citations.
It makes graph-derived context inspectable and reproducible.
It is distinctive enough to make the database proposal meaningful.
```

Limitations:

```text
Public vector search API must be verified.
Variable-length traversal may need application-side BFS until native k-hop APIs are available.
Benchmark reports must distinguish native execution from adapter fallback execution.
```

Reason for selecting caracaldb:

```text
The project is specifically about proposing caracaldb as the GraphRAG database.
It fits a local, inspectable, graph-aware RAG implementation.
It allows the system to demonstrate both structured storage and graph-style retrieval.
```

### Neo4j only

Neo4j is the selected comparison Graph DB for Config 2.

Advantages:

```text
It is highly suitable for entity and relationship traversal.
Cypher is expressive for graph queries.
It fits multi-hop GraphRAG expansion naturally.
Neo4j 5.x can support native vector indexes.
```

Limitations:

```text
It introduces a separate database server and query language.
Document chunk and citation storage may require additional modeling decisions.
Large Arrow-native analytical exports are less natural than in an embedded analytical database.
```

Reason for using Neo4j as comparison:

```text
Neo4j gives a strong graph-native upper-bound comparison.
It keeps the comparison focused on Graph DB only architecture rather than generic RDB baselines.
```

### CaracalDB + VectorDB

Config 3 keeps CaracalDB as the authoritative graph ecosystem while using a VectorDB as an external semantic index.

Advantages:

```text
It makes semantic candidate generation stronger and more conventional.
It keeps CaracalDB responsible for graph context, semantic candidate re-entry, citations, and inspectable state.
It demonstrates a realistic hybrid GraphRAG architecture.
```

Limitations:

```text
It introduces cross-store synchronization.
Chunk IDs and embedding IDs must stay consistent across CaracalDB and VectorDB.
Benchmark results must include the cost of mapping external semantic hits back into graph nodes and retrieval traces.
```

Reason for including this configuration:

```text
It shows whether adding an external semantic index improves retrieval enough to justify extra complexity.
It positions CaracalDB as the durable GraphRAG state database.
```

### Neo4j + VectorDB

Config 4 combines graph-native traversal with dedicated vector retrieval.

Advantages:

```text
It represents a best-of-breed GraphRAG architecture.
Neo4j handles relation topology while VectorDB handles external semantic candidate generation.
It is a strong external comparison target.
```

Limitations:

```text
It has the highest operational complexity among the four configurations.
It requires cross-store joins and synchronization.
It is less suitable as a lightweight embedded local architecture.
```

Reason for including this configuration:

```text
It provides a strong comparison against the CaracalDB proposal.
It helps separate retrieval quality gains from operational complexity costs.
```

## 3.8 Final Technology Selection

The project uses:

```text
Primary proposal: CaracalDB only
Extended proposal: CaracalDB + VectorDB
Comparison baseline: Neo4j only
Extended comparison baseline: Neo4j + VectorDB
Retrieval Pattern: GraphRAG
Programming Language: Python
```

Selection rationale:

```text
The selected stack directly supports the project goal of proposing caracaldb.
CaracalDB only tests whether an embedded analytical GraphDB can support the full GraphRAG process.
CaracalDB + VectorDB tests whether an external semantic index improves the proposal while still re-entering results into the graph ecosystem.
Neo4j only provides a strong graph-native single-database comparison.
Neo4j + VectorDB provides a best-of-breed comparison with higher operational complexity.
The project demonstrates where DB-core features end and GraphRAG adapter logic begins.
```

# 4. Comparison System

## 4.1 Purpose

The comparison system explains and measures why `caracaldb` is proposed for this GraphRAG project compared with graph-native and vector-assisted alternatives.

The comparison must evaluate four configurations:

```text
Config 1: CaracalDB only
Config 2: Neo4j only
Config 3: CaracalDB + VectorDB
Config 4: Neo4j + VectorDB
```

The benchmark should be runnable where possible. Neo4j may be skipped when local Docker or connection settings are unavailable. VectorDB configurations may use Chroma by default or FAISS as an external semantic index fallback.

## 4.2 Comparison Workload

The benchmark must use a GraphRAG-style workload, not only a single text lookup.

Required workload:

```text
Load documents and chunks
Create semantic entry candidates for a question
Map semantic candidates to Chunk and Entity nodes
Match question terms to graph entities
Expand relation topology around matched and semantic candidate entities
Retrieve evidence chunks through relation and provenance paths
Generate or simulate answer citations
Persist retrieval trace
Measure latency and result counts
```

Graph DB only workload requirements:

```text
The single Graph DB must store graph-connected artifacts, not detached tables.
The single Graph DB must expose embeddings as graph-addressable semantic entry points.
The single Graph DB must represent semantic similarity as vector neighborhoods, semantic buckets, or SEMANTIC_NEIGHBOR edges.
The single Graph DB must represent relationships as typed, weighted, traversable relation topology.
The benchmark must report which semantic entry and relation expansion strategy was used.
```

Recommended default:

```text
Benchmark question: "Why is GraphRAG useful for grounded question answering?"
Retrieval modes: semantic_neighborhood, relation_topology, graph_ecosystem
Top-k semantic candidates: 8
Relation depth: 2
Workload repeats: 10
```

For each configuration, the benchmark should report:

```text
config_id
config_name
graph_db
vector_db
status
total_seconds
load_seconds
semantic_entry_seconds
semantic_reentry_seconds
relation_expansion_seconds
trace_write_seconds
documents
chunks
entities
relationships
retrieved_context_items
citations
adapter_loc
semantic_entry_mode
semantic_reentry_mode
relation_expand_mode
cross_store_join_count
sync_notes
notes
```

## 4.3 Example Comparison Output

The comparison system may be implemented as:

```text
src/compare_configs.py
```

Example output:

```text
# GraphRAG Configuration Comparison

Each cell is:
status / total_seconds / context_items / semantic_entry / relation_expand

| Config | Architecture | Status | Total Seconds | Context Items | Semantic Entry | Relation Expand | Notes |
| ------ | ------------ | -----: | ------------: | ------------: | ----------- | ----------------- | ----- |
| 1 | CaracalDB only | ok | 1.40 | 42 | caracal_hnsw or semantic_neighbor_edges | tuft_paths or bfs_fallback | proposed graph ecosystem |
| 2 | Neo4j only | ok/skipped | 1.70 | 42 | neo4j_vector_index | cypher_paths | graph-native comparison |
| 3 | CaracalDB + Chroma | ok | 1.25 | 44 | chroma_to_graph_candidates | tuft_paths or bfs_fallback | external semantic index re-enters CaracalDB |
| 4 | Neo4j + Chroma | ok/skipped | 1.55 | 44 | chroma_to_graph_candidates | cypher_paths | external semantic index re-enters Neo4j |
```

The system should export:

```text
outputs/comparison_report.txt
outputs/comparison_benchmark.csv
outputs/database_proposal.md
```

## 4.4 Benchmark Fairness Rules

The benchmark must follow these fairness rules:

```text
All four configurations use the same documents, questions, chunks, entities, and relationships.
All configurations use the same embedding model or deterministic mock embeddings.
All configurations use the same top_k_semantic_candidates, relation_depth, and scoring weights.
Graph DB only configurations may use native vector index, exact scan, semantic buckets, or semantic-neighborhood edges, but the chosen strategy must be reported.
VectorDB configurations must include external semantic index lookup, graph-node re-entry, and synchronization overhead in timing.
Skipped external systems must include a clear reason.
Adapter fallback code must not be hidden as native database capability.
External semantic hits must be mapped back into graph nodes before relation expansion and answer grounding.
```

## 4.5 Evaluation Criteria

From a client perspective, the project can be evaluated using the following criteria:

```text
Whether caracaldb is clearly proposed and used as the main database
Whether the four comparison configurations are implemented or clearly skipped
Whether Graph DB only configurations are optimized fairly
Whether CaracalDB native behavior is separated from adapter fallback behavior
Whether documents are chunked and stored correctly
Whether entities and relationships are extracted and validated
Whether GraphRAG retrieval combines semantic neighborhoods, entities, relation topology, evidence paths, and citations
Whether answer generation is grounded in retrieved context
Whether retrieval traces are exported and inspectable
Whether the configuration comparison is fair and practical
Whether external database dependencies are skipped clearly when unavailable
Whether README and execution instructions are clear
Whether tests or validation results are included
```

# 5. Suggested Implementation Milestones

## Milestone 1: Local GraphRAG Skeleton

```text
Create project structure
Load sample documents
Implement chunking
Write processed chunks to CSV
Add basic tests
```

## Milestone 2: caracaldb Storage

```text
Define storage schema
Load documents and chunks into caracaldb
Implement storage adapter
Validate row counts
Verify public APIs for vector index/search
Verify available traversal APIs for k-hop style expansion
```

## Milestone 3: Graph Construction

```text
Extract entities
Normalize entity names
Extract relationships
Store chunk-entity mentions
Store entity relationships
```

## Milestone 4: Retrieval

```text
Implement semantic_neighborhood retrieval
Implement relation_topology retrieval
Implement graph_ecosystem retrieval
Implement CaracalDB native vector search if API is available
Implement Arrow/numpy vector fallback if native API is unavailable
Implement Tuft relation expansion where possible
Implement Python BFS fallback for missing variable-length traversal
Export retrieval traces
```

## Milestone 5: Answering and Evaluation

```text
Generate grounded answers
Attach citations
Run evaluation questions
Export answer logs and evaluation reports
```

## Milestone 6: Database Proposal

```text
Implement four storage adapters
Run Config 1 CaracalDB only benchmark
Run Config 2 Neo4j only benchmark when available
Run Config 3 CaracalDB + VectorDB benchmark
Run Config 4 Neo4j + VectorDB benchmark when available
Mark unavailable external systems as skipped with clear reasons
Report native vs fallback execution modes
Write database proposal report
Finalize README
```
