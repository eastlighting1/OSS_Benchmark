# KGMovie

Knowledge graph-based movie exploration system for the TMDB 5000 dataset.

## Task Requirements

The task was to implement a toy knowledge graph system for movie data using
the required libraries `lynxes` and `caracaldb`.

The system must:

- Load `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`.
- Use `lynxes` in the DataFrame/ETL layer.
- Clean and merge movie and credit data by movie ID.
- Parse JSON-like columns for genres, keywords, cast, and crew.
- Extract `Movie`, `Person`, `Genre`, and `Keyword` entities.
- Extract `ACTED_IN`, `DIRECTED`, `HAS_GENRE`, and `HAS_KEYWORD` relationships.
- Store the graph-shaped tables in `caracaldb`.
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

- DataFrame/ETL: `lynxes`
- Storage/query: `caracaldb`

This combination satisfies the required stack and is the primary implementation
used by the project pipeline.

The comparison matrix evaluates the proposed stack against common alternatives:

| Dimension | Proposed | Comparison libraries |
| --- | --- | --- |
| DataFrame/ETL | `lynxes` | pandas, Polars, Dask DataFrame |
| Database/query | `caracaldb` | SQLite, PostgreSQL, Neo4j |

The latest benchmark includes all 16 combinations:

```text
DataFrame/ETL: lynxes, pandas, Polars, Dask
Database/query: caracaldb, SQLite, PostgreSQL, Neo4j
Status: 16 ok / 16 total
Rows checked per cell: 119700 aggregate KG workload rows
```

The benchmark runner validates correctness before treating a cell as `ok`.
For each DataFrame result, it computes the expected aggregate KG workload count
once, then every database adapter must return the same count. If a backend runs
a weaker query or returns a different result count, that cell is marked
`failed` instead of being included as a successful benchmark.

## Result Comparison and Analysis

Latest local benchmark result:

| DataFrame \ Database | caracaldb | SQLite | PostgreSQL | Neo4j |
| --- | ---: | ---: | ---: | ---: |
| lynxes | 1.1499s | 6.1883s | 10.7871s | 3.8606s |
| pandas | 1.4599s | 6.5020s | 10.9969s | 3.2596s |
| Polars | 1.2076s | 6.1744s | 10.6793s | 2.8294s |
| Dask | 1.5670s | 6.5618s | 10.9853s | 3.2167s |

The proposed `lynxes + caracaldb` stack is the fastest full combination in the
latest run:

```text
lynxes + caracaldb: 1.1499s total
Polars + caracaldb: 1.2076s total
pandas + caracaldb: 1.4599s total
Dask + caracaldb:   1.5670s total
```

Key observations:

- `caracaldb` is the fastest database/query backend for this local toy KG
  workload. Its full-load plus indexed workload path completes in about 0.7s
  after the DataFrame stage.
- `lynxes` is the fastest DataFrame/ETL path in this run. The final ingestion
  path uses `lynxes.read_csv` with PyArrow engine and column projection to avoid
  materializing unused TMDB columns.
- Polars is a close DataFrame baseline, but it does not beat the required
  `lynxes + caracaldb` combination end to end in the latest 4x4 result.
- Neo4j is graph-native and expressive, and it beats the SQL backends in this
  run, but its external-service setup and load/query cycle are still heavier
  than `caracaldb` for the winning end-to-end combination.
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
uv run python main.py compare
```

To include PostgreSQL and Neo4j in the 4x4 matrix, start both services and set:

```powershell
$env:POSTGRES_DSN="postgresql://kg:kgpass@localhost:5432/kgbench"
$env:NEO4J_URI="bolt://localhost:7687"
$env:NEO4J_USER="neo4j"
$env:NEO4J_PASSWORD="kgpass123"
uv run python main.py compare
```

The latest full run used all four database backends and all four DataFrame/ETL
paths. Every cell produced `4803` movies and `119700` aggregate KG workload
rows, so the timing comparison is based on equivalent results rather than
different workloads.

## Project Layout

```text
TASK 1/
  data/processed/          Generated normalized CSV tables and caracaldb bundle
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
6. Stores the graph-shaped data in `caracaldb`.
7. Builds a `lynxes` graph frame and writes graph summary output.
8. Generates CSV reports.

## Ingestion Decision

The final ingestion path uses the released `lynxes.read_csv` API with column
projection:

```text
lynxes.read_csv(..., engine="pyarrow", convert_options=ConvertOptions(include_columns=required_columns))
```

This keeps CSV ingestion inside `lynxes`, avoids the old project-local CSV
shim, and prevents unused TMDB columns from being materialized into Python row
dictionaries. The `caracaldb` load path uses the released Arrow bulk insert
APIs: `insert_node_table_arrow` and `insert_edge_table_arrow`.

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

The query commands read back from the generated `caracaldb` bundle rather than
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
outputs/lynxes_graph_summary.txt
outputs/tmdb_kg.gf
```

See `outputs/README.md` for a file-by-file explanation of each generated
artifact, its purpose, and whether it is appropriate to publish in a public
repository.

`compare` runs a real 4x4 comparison matrix. Each DataFrame/ETL implementation
is crossed with each database/query implementation:

- DataFrame/ETL: `lynxes`, pandas, Polars, Dask DataFrame
- Database/query: `caracaldb`, SQLite, PostgreSQL, Neo4j

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

The pipeline rebuilds the local `caracaldb` bundle on each full run. This keeps repeated executions reproducible and prevents accidental duplicate storage.
