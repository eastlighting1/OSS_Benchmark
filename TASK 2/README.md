# GraphRAG CaracalDB Benchmark

Task 2 benchmark implementation for comparing:

```text
Config 1: CaracalDB only
Config 2: Neo4j only
Config 3: CaracalDB + Chroma external semantic index
Config 4: Neo4j + Chroma external semantic index
```

The retriever now includes query entity linking, question-type retrieval planning, multi-hop evidence path expansion, answer-aware context reranking, evidence-grounded answer span extraction, citation reranking, and gold-answer correctness metrics.

## Setup

Python dependencies are managed by `uv`:

```bash
uv sync --upgrade
```

Neo4j is optional but required to execute Config 2 and Config 4:

```powershell
docker run --name task2-neo4j `
  -p 7474:7474 -p 7687:7687 `
  -e NEO4J_AUTH=neo4j/password `
  -d neo4j:5-community

$env:NEO4J_URI="bolt://localhost:7687"
$env:NEO4J_USER="neo4j"
$env:NEO4J_PASSWORD="password"
```

Without Neo4j, those configurations are reported as `skipped`.

## Run

```bash
uv run python main.py compare-configs
```

MultiHopRAG mode uses `../data/MultiHopRAG.json` and `../data/corpus.json` by default:

```bash
uv run python main.py compare-configs --dataset multihoprag --max-questions 25 --max-documents 200 --repeats 1 --embedding-dimension 256
```

Outputs are written to:

```text
outputs/answer_log.md
outputs/comparison_benchmark.csv
outputs/comparison_report.txt
outputs/database_proposal.md
outputs/evaluation_report.csv
outputs/retrieval_trace.json
```

`evaluation_report.csv` includes document-level evidence recall plus answer metrics:

```text
retrieval_precision_at_k
context_recall
citation_recall
answer_exact_match
answer_contains_gold
answer_token_f1
```

Tests:

```bash
uv run pytest
```
