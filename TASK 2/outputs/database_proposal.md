# CaracalDB GraphRAG Database Proposal

## Recommendation

Use CaracalDB as the authoritative graph ecosystem database for Task 2.
CaracalDB stores source nodes, chunk nodes, entity nodes, embeddings, relation topology, evidence paths, retrieval traces, answers, and citations.

## Current Benchmark Signal

- CaracalDB only: status=ok, semantic=exact_scan, relation=memory_bfs_fallback, total=147.90361799974926
- CaracalDB + external semantic index: status=ok, semantic=chroma_persistent, relation=memory_bfs_fallback, total=79.28788599953987

## Interpretation

The CaracalDB-only path validates the central proposal when it uses graph-addressable HNSW semantic entry and native relation expansion.
The external semantic index path is useful when semantic candidate generation should be delegated to a specialized index, but its results must re-enter the graph before relation expansion and answer grounding.

## Remaining External Dependency

Neo4j comparisons require a running Neo4j server configured through NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD.
