# KGMovie

Knowledge graph-based movie exploration system for the TMDB 5000 dataset.

## Task Requirements

The task was to implement a toy knowledge graph system for movie data using
the required libraries `Lynxes` and `CaracalDB`.

The system must:

- Load `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`.
- Use `Lynxes` in the DataFrame/ETL layer.
- Clean and merge movie and credit data by movie ID.
- Parse JSON-like columns for genres, keywords, cast, and crew.
- Extract `Movie`, `Person`, `Genre`, and `Keyword` entities.
- Extract `ACTED_IN`, `DIRECTED`, `HAS_GENRE`, and `HAS_KEYWORD` relationships.
- Store the graph-shaped tables in `CaracalDB`.
- Provide CLI queries for movie lookup, actor/director lookup, genre search, repeated director-actor collaboration analysis, central-person analysis, and similar movie recommendation.
- Export processed tables, reports, and basic test coverage.
- Produce a real 4x4 benchmark comparing DataFrame/ETL choices and
  database/query choices with a knowledge graph-style workload.

The benchmark workload is intentionally more than a single relational lookup.
Each runnable cell repeats the following KG tasks 25 times:

- Science Fiction genre search
- Tom Hanks actor-to-movie lookup
- Christopher Nolan director-to-movie lookup
- Repeated director-actor collaboration count
- Interstellar-style recommendation overlap count

## Proposed Stack and Comparison Libraries

The proposed library combination is:

- DataFrame/ETL: `Lynxes`
- Storage/query: `CaracalDB`

This combination satisfies the required stack and is the primary implementation
used by the project pipeline.

The comparison matrix evaluates the proposed stack against common alternatives:

| Dimension | Proposed | Comparison libraries |
| --- | --- | --- |
| DataFrame/ETL | `Lynxes` | pandas, Polars, Dask DataFrame |
| Database/query | `CaracalDB` | SQLite, PostgreSQL, Neo4j |

The latest benchmark includes all 16 combinations:

```text
DataFrame/ETL: Lynxes, pandas, Polars, Dask
Database/query: CaracalDB, SQLite, PostgreSQL, Neo4j
Status: 16 ok / 16 total
Rows checked per cell: 119700 aggregate KG workload rows
Timing policy: median of 3 verified runs
```

The benchmark runner validates correctness before treating a cell as `ok`.
For each DataFrame result, it computes the expected aggregate KG workload count
once, then every database adapter must return the same count. If a backend runs
a weaker query or returns a different result count, that cell is marked
`failed` instead of being included as a successful benchmark.

## Result Comparison and Analysis

Latest local benchmark result:

| DataFrame \ Database | CaracalDB | SQLite | PostgreSQL | Neo4j |
| --- | ---: | ---: | ---: | ---: |
| Lynxes | 1.7718s | 6.1967s | 10.5062s | 2.7774s |
| pandas | 2.0896s | 6.4876s | 10.8771s | 5.2104s |
| Polars | 1.8849s | 6.2815s | 10.6925s | 5.2295s |
| Dask | 2.2196s | 6.5647s | 10.9551s | 5.8202s |

The proposed `Lynxes + CaracalDB` stack is the fastest full combination in the
latest run:

```text
Lynxes + CaracalDB: 1.7718s total
Polars + CaracalDB: 1.8849s total
pandas + CaracalDB: 2.0896s total
Dask + CaracalDB:   2.2196s total
```

Key observations:

- `CaracalDB` is the fastest database/query backend for this local toy KG
  workload when compared with the same `Lynxes` DataFrame stage. Its full-load
  plus native graph API workload path completes in about 1.34s after the
  DataFrame stage.
- `Lynxes` is the fastest DataFrame/ETL path in this run. The final ingestion
  path uses `Lynxes.read_csv(columns=...)` and `NodeFrame.to_rows()` to keep CSV
  projection and row materialization inside the library API.
- Polars is a close DataFrame baseline, but it does not beat the required
  `Lynxes + CaracalDB` combination end to end in the latest 4x4 result.
- Neo4j is graph-native and expressive, and it beats the SQL backends in this
  run, but its external-service setup and load/query cycle are still heavier
  than `CaracalDB` when paired with the winning `Lynxes` ETL path.
- SQLite is easy to run but slower for the repeated KG-style workload.
- PostgreSQL is robust, but in this benchmark it has the highest total time
  because the test uses temporary table loading plus repeated SQL queries.

The detailed generated files are:

- `outputs/comparison_report.txt`
- `outputs/comparison_benchmark.csv`
- `outputs/README.md`

## Benchmark Command

Run the local benchmark without external services:

```bash
uv run python main.py compare --benchmark-runs 3
```

To include PostgreSQL and Neo4j in the 4x4 matrix, start both services and set:

```powershell
$env:POSTGRES_DSN="postgresql://kg:kgpass@localhost:5432/kgbench"
$env:NEO4J_URI="bolt://localhost:7687"
$env:NEO4J_USER="neo4j"
$env:NEO4J_PASSWORD="kgpass123"
uv run python main.py compare --benchmark-runs 3
```

The latest full run used all four database backends and all four DataFrame/ETL
paths. Every cell produced `4803` movies and `119700` aggregate KG workload
rows, so the timing comparison is based on equivalent results rather than
different workloads.

## Project Layout

```text
TASK 1/
  data/processed/          Generated normalized CSV tables and CaracalDB bundle
  outputs/                 Generated reports and graph artifacts
  src/                     ETL, schema, DB loading, query, recommender modules
  tests/                   Unit tests for transforms, schema, queries
  main.py                  CLI entry point
```

The raw CSV files are expected one directory above this task folder:

```text
../data/tmdb_5000_movies.csv
../data/tmdb_5000_credits.csv
```

You can also pass explicit paths with `--movies` and `--credits`.

## Setup

```bash
uv sync
```

or run commands directly with `uv run`, which will create the environment.

## Full Pipeline

```bash
uv run python main.py
```

Equivalent with explicit input paths:

```bash
uv run python main.py --movies ../data/tmdb_5000_movies.csv --credits ../data/tmdb_5000_credits.csv
```

The pipeline:

1. Loads the TMDB CSV files.
2. Cleans and merges movie and credit rows by movie ID.
3. Parses genres, keywords, cast, and crew JSON-like columns.
4. Extracts `Movie`, `Person`, `Genre`, and `Keyword` entities.
5. Extracts `ACTED_IN`, `DIRECTED`, `HAS_GENRE`, and `HAS_KEYWORD` relationships.
6. Stores the graph-shaped data in `CaracalDB`.
7. Builds a `Lynxes` graph frame and writes graph summary output.
8. Generates CSV reports.

## Ingestion Decision

The final ingestion path uses released `Lynxes` APIs for column projection and
row materialization:

```text
Lynxes.read_csv(..., columns=required_columns, schema_overrides=...)
NodeFrame.to_rows()
```

This keeps CSV ingestion inside `Lynxes`, avoids the old project-local CSV
shim, and prevents unused TMDB columns from being materialized into Python row
dictionaries. The `CaracalDB` path uses released Arrow bulk insert and graph
query APIs: `insert_node_table_arrow`, `insert_edge_table_arrow`, `nodes`,
`in_`, `out`, `node_table`, and `edge_table`.

## CLI Commands

```bash
uv run python main.py query-collaborations --limit 20 --min-count 2
uv run python main.py query-collaborations --director "Christopher Nolan" --limit 10
uv run python main.py query-movie --movie "Interstellar"
uv run python main.py query-actor --actor "Tom Hanks"
uv run python main.py query-director --director "Christopher Nolan"
uv run python main.py query-genre --genres "Science Fiction"
uv run python main.py query-genre --genres "Action,Thriller"
uv run python main.py recommend --movie "Interstellar" --limit 10
uv run python main.py compare
```

The query commands read back from the generated `CaracalDB` bundle rather than
from the processed CSV files. The processed CSV files are kept as inspectable
reports of the normalized tables.

## Generated Outputs

After running the full pipeline:

```text
data/processed/movies.csv
data/processed/persons.csv
data/processed/genres.csv
data/processed/keywords.csv
data/processed/acted_in.csv
data/processed/directed.csv
data/processed/movie_genres.csv
data/processed/movie_keywords.csv
data/processed/tmdb_kg.crcl/
outputs/collaboration_report.csv
outputs/genre_report.csv
outputs/recommendation_report.csv
outputs/central_people_report.csv
outputs/comparison_report.txt
outputs/comparison_benchmark.csv
outputs/README.md
outputs/Lynxes_graph_summary.txt
outputs/tmdb_kg.gf
```

See `outputs/README.md` for a file-by-file explanation of each generated
artifact, its purpose, and whether it is appropriate to publish in a public
repository.

`compare` runs a real 4x4 comparison matrix. Each DataFrame/ETL implementation
is crossed with each database/query implementation:

- DataFrame/ETL: `Lynxes`, pandas, Polars, Dask DataFrame
- Database/query: `CaracalDB`, SQLite, PostgreSQL, Neo4j

PostgreSQL runs when `POSTGRES_DSN` is configured. Neo4j runs when `NEO4J_URI`,
`NEO4J_USER`, and `NEO4J_PASSWORD` are configured. Without those services, their
matrix cells are reported as `skipped` while still showing adapter LOC and setup
notes. The comparison report includes elapsed time, approximate adapter code
length, aggregate KG workload row counts, status, and setup notes for every
matrix cell. The DB phase runs repeated graph-style queries, not just a single
relational genre lookup.

Sample pipeline counts from the provided dataset:

```text
movies: 4803
persons: 11718
genres: 20
keywords: 9812
acted_in: 23589
directed: 5166
movie_genres: 12160
movie_keywords: 36192
```

## Tests

```bash
uv run pytest
```

Current verification result:

```text
9 passed
```

## Notes

The pipeline rebuilds the local `CaracalDB` bundle on each full run. This keeps repeated executions reproducible and prevents accidental duplicate storage.
