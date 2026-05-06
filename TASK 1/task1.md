# Knowledge Graph-Based Movie Data Exploration System

## 1. Implementation Specification

### 1.1 Project Overview

This project aims to build a toy knowledge graph system using publicly available movie data. The system should model relationships among movies, actors, directors, genres, and keywords, and provide basic exploration, collaboration analysis, and movie recommendation features.

The data processing layer must use `lynxes` as the DataFrame library, while the storage and query layer must use `caracaldb`.

The purpose of this project is not to build a production-grade service, but to demonstrate data modeling, ETL design, database loading, relationship querying, and system comparison skills.

### 1.2 Dataset

The recommended dataset is:

```text
TMDB 5000 Movie Dataset
```

The system assumes the following input files:

```text
tmdb_5000_movies.csv
tmdb_5000_credits.csv
```

The raw data is expected to include:

```text
Movie ID
Movie title
Release date
Budget
Revenue
Overview
Genres
Keywords
Cast
Crew
Director information
```

### 1.3 Technology Stack

The system must use the following technologies:

```text
Programming Language: Python
Database: caracaldb
DataFrame Library: lynxes
Data Source Format: CSV
Output Format: CLI output, CSV reports, or JSON reports
```

`caracaldb` and `lynxes` are mandatory components. `lynxes` must be used for the data processing layer, and `caracaldb` must be used for the database storage layer.

### 1.4 System Architecture

The system should follow this pipeline:

```text
1. Load raw data
2. Clean and transform data
3. Extract entities
4. Extract relationships
5. Store data in caracaldb
6. Execute knowledge graph-style queries
7. Generate analysis results
8. Produce a real 4x4 comparison benchmark output
```

Overall flow:

```text
CSV Dataset
   ->
lynxes DataFrame
   ->
Entity / Relationship Extraction
   ->
caracaldb
   ->
Knowledge Graph Query Layer
   ->
Analysis / Recommendation Output
```

### 1.5 Data Model

The system must include at least the following entities:

```text
Movie
Person
Genre
Keyword
```

Entity meanings:

```text
Movie: Core entity representing a movie
Person: Actor, director, or other movie-related person
Genre: Movie genre
Keyword: Keyword or tag describing a movie
```

### 1.6 Entity Schema

Movie:

```text
movie_id: integer
title: string
release_date: date
budget: integer
revenue: integer
overview: string
```

Person:

```text
person_id: integer
name: string
```

Genre:

```text
genre_id: integer
name: string
```

Keyword:

```text
keyword_id: integer
name: string
```

### 1.7 Relationship Model

The system must support at least the following relationships:

```text
Person - ACTED_IN - Movie
Person - DIRECTED - Movie
Movie - HAS_GENRE - Genre
Movie - HAS_KEYWORD - Keyword
```

Relationship meanings:

```text
ACTED_IN: A person appeared in a movie as an actor
DIRECTED: A person directed a movie
HAS_GENRE: A movie belongs to a genre
HAS_KEYWORD: A movie has a keyword
```

### 1.8 caracaldb Storage Structure

The system should store entities and relationships separately in `caracaldb`.

Required storage objects:

```text
movies
persons
genres
keywords
acted_in
directed
movie_genres
movie_keywords
```

`movies`:

```text
movie_id
title
release_date
budget
revenue
overview
```

`persons`:

```text
person_id
name
```

`genres`:

```text
genre_id
name
```

`keywords`:

```text
keyword_id
name
```

`acted_in`:

```text
person_id
movie_id
character_name
cast_order
```

`directed`:

```text
person_id
movie_id
```

`movie_genres`:

```text
movie_id
genre_id
```

`movie_keywords`:

```text
movie_id
keyword_id
```

### 1.9 lynxes DataFrame Processing Specification

`lynxes` must be used for the following tasks:

```text
Load CSV files
Handle missing values
Remove duplicates
Parse JSON-like string columns
Merge movie data and credit data
Create entity tables
Create relationship tables
Create analysis result tables
```

Processing rules:

```text
CSV loading must use the official lynxes.read_csv API.
The implementation may use lynxes reader engine/projection options to read only required columns.
Project-local CSV shims should be fallback or test helpers, not the primary ingestion path.
Movie data and credit data must be merged using the movie ID.
Genre, keyword, cast, and crew columns must be normalized into row-level data.
Only the top N actors per movie may be used; the default value is 5.
Directors must be extracted from crew records where job equals Director.
Duplicate entities must be removed.
Relationship tables must not contain references to non-existing entity IDs.
```

### 1.10 Main Features

#### Feature 1: Data Ingestion

The system must read raw CSV files, convert them into `lynxes` DataFrames, clean the data, and store the processed result in `caracaldb`.

Input:

```text
tmdb_5000_movies.csv
tmdb_5000_credits.csv
```

Output:

```text
Entity and relationship tables stored in caracaldb
```

#### Feature 2: Director-Actor Collaboration Analysis

The system must identify actors who have collaborated multiple times with the same director.

Example queries:

```text
Find the top 20 repeated director-actor collaborations
Find the actors who worked most frequently with a given director
```

Expected output:

```text
director_name
actor_name
collaboration_count
movie_titles
```

#### Feature 3: Genre-Based Movie Search

The system must allow users to search for movies by genre.

Example queries:

```text
Find all Science Fiction movies
Find movies that are both Action and Thriller
```

Expected output:

```text
movie_id
title
genres
release_date
revenue
```

#### Feature 4: Similar Movie Recommendation

Given a movie, the system must recommend similar movies using genre, keyword, actor, and director information.

Similarity factors:

```text
Number of shared genres
Number of shared keywords
Number of shared actors
Whether the director is the same
```

Default scoring formula:

```text
similarity_score =
    2.0 * common_genre_count
  + 1.5 * common_keyword_count
  + 2.0 * common_actor_count
  + 3.0 * same_director_flag
```

Expected output:

```text
base_movie
recommended_movie
similarity_score
common_genres
common_keywords
common_actors
same_director
```

#### Feature 5: Central Person Analysis

The system must identify highly connected people in the knowledge graph.

The analysis should include at least one of the following metrics:

```text
Number of acted movies
Number of directed movies
Genre diversity
Number of unique collaborators
Connection centrality
```

Expected output:

```text
person_name
acted_movie_count
directed_movie_count
unique_collaborator_count
centrality_score
```

### 1.11 Recommended Module Structure

```text
movie_kg_project/
├── data/
│   ├── raw/
│   │   ├── tmdb_5000_movies.csv
│   │   └── tmdb_5000_credits.csv
│   └── processed/
│       ├── movies.csv
│       ├── persons.csv
│       ├── genres.csv
│       ├── keywords.csv
│       ├── acted_in.csv
│       ├── directed.csv
│       ├── movie_genres.csv
│       ├── movie_keywords.csv
│       └── tmdb_kg.crcl/
├── src/
│   ├── config.py
│   ├── ingest.py
│   ├── transform.py
│   ├── schema.py
│   ├── load_to_db.py
│   ├── storage.py
│   ├── db_queries.py
│   ├── queries.py
│   ├── recommender.py
│   ├── kg.py
│   ├── pipeline.py
│   └── compare_systems.py
├── outputs/
│   ├── collaboration_report.csv
│   ├── genre_report.csv
│   ├── recommendation_report.csv
│   ├── central_people_report.csv
│   ├── comparison_report.txt
│   ├── comparison_benchmark.csv
│   ├── lynxes_graph_summary.txt
│   └── tmdb_kg.gf
├── tests/
│   ├── test_transform.py
│   ├── test_schema.py
│   └── test_queries.py
├── README.md
└── main.py
```

### 1.12 Execution

The full pipeline should be executable with:

```bash
python main.py --movies data/raw/tmdb_5000_movies.csv --credits data/raw/tmdb_5000_credits.csv
```

Optional commands may include:

```bash
python main.py ingest
python main.py build-kg
python main.py query-movie --movie "Interstellar"
python main.py query-actor --actor "Tom Hanks"
python main.py query-director --director "Christopher Nolan"
python main.py query-collaborations
python main.py query-collaborations --director "Christopher Nolan"
python main.py query-genre --genres "Action,Thriller"
python main.py recommend --movie "Interstellar"
python main.py compare
```

### 1.13 Deliverables

The final submission must include:

```text
Source code
README with setup and execution instructions
caracaldb data files or database generation scripts
lynxes-based data transformation code
Sample query results
4x4 comparison benchmark report
Comparison benchmark CSV
Basic tests
```

# 2. Requirements

## 2.1 Functional Requirements

### FR-001. Data Loading

The system shall load the provided CSV files into `lynxes` DataFrames.

Acceptance criteria:

```text
The system can read both the movies CSV and credits CSV.
File paths are provided through configuration or CLI arguments.
If loading fails, the system returns a clear error message.
```

### FR-002. Data Cleaning

The system shall clean missing, duplicated, or invalid values.

Acceptance criteria:

```text
Rows without required IDs are removed.
Rows without movie titles are removed.
Duplicate movie_id records are removed.
budget and revenue values are converted to numeric types.
```

### FR-003. Entity Extraction

The system shall extract movie, person, genre, and keyword entities.

Acceptance criteria:

```text
The movies table contains one row per movie.
The persons table stores actors and directors without duplicates.
The genres table stores genres without duplicates.
The keywords table stores keywords without duplicates.
```

### FR-004. Relationship Extraction

The system shall extract relationships between entities.

Acceptance criteria:

```text
acted_in stores actor-to-movie relationships.
directed stores director-to-movie relationships.
movie_genres stores movie-to-genre relationships.
movie_keywords stores movie-to-keyword relationships.
```

### FR-005. caracaldb Storage

The system shall store all processed entities and relationships in `caracaldb`.

Acceptance criteria:

```text
All required entity and relationship tables are created.
The number of stored rows matches the number of transformed rows.
Repeated pipeline execution does not create unintended duplicate records.
```

### FR-006. Basic Query Support

The system shall support basic queries against the stored data.

Acceptance criteria:

```text
The system can retrieve movie information by title.
The system can retrieve movies by actor.
The system can retrieve movies by director.
The system can retrieve movies by genre.
```

### FR-007. Collaboration Analysis

The system shall analyze repeated director-actor collaborations.

Acceptance criteria:

```text
The system calculates collaboration counts for director-actor pairs.
Results are sorted by collaboration count in descending order.
A minimum collaboration threshold can be configured.
```

### FR-008. Similar Movie Recommendation

The system shall recommend movies similar to a given movie.

Acceptance criteria:

```text
The input movie can be specified by title or movie_id.
The recommendation result does not include the input movie itself.
The result includes the reasoning behind the similarity score.
The user can request the top N recommendations.
```

### FR-009. Report Generation

The system shall export analysis results as files.

Acceptance criteria:

```text
Collaboration analysis results can be exported as CSV.
Genre-based movie statistics can be exported as CSV.
Recommendation results can be exported as CSV or JSON.
```

### FR-010. Comparison System Output

The system shall provide a runnable 4x4 comparison matrix between DataFrame/ETL tools and database/query tools.

Acceptance criteria:

```text
The DataFrame/ETL dimension must include lynxes, pandas, Polars, and Dask DataFrame.
The database/query dimension must include caracaldb, SQLite, PostgreSQL, and Neo4j.
Each DataFrame option must be crossed with each database option, producing 16 matrix cells.
Each runnable cell must measure elapsed time, approximate adapter code length, result row count, status, and notes.
The benchmark must use knowledge graph-style workload, not only a single relational lookup.
The KG workload must include genre search, actor movie lookup, director movie lookup, director-actor collaboration analysis, and recommendation-style overlap analysis.
PostgreSQL and Neo4j may be skipped when external connection settings are not configured, but executable adapters must be implemented.
The comparison must be exported as a human-readable report and a CSV file.
```

## 2.2 Non-Functional Requirements

### NFR-001. Reproducibility

Given the same input data and configuration, the system shall produce the same output.

### NFR-002. Modularity

Data loading, transformation, storage, querying, recommendation, and comparison logic shall be separated into independent modules.

### NFR-003. Extensibility

The system shall be designed so that the following entities can be added later:

```text
ProductionCompany
Country
Language
Review
Rating
```

### NFR-004. Maintainability

Functions and classes shall use clear names. Core logic shall include comments or docstrings.

### NFR-005. Error Handling

The system shall handle the following cases:

```text
Missing input file
CSV schema mismatch
Unparseable JSON-like string
Database connection failure
Empty query result
Missing target movie for recommendation
```

### NFR-006. Performance

For the TMDB 5000 dataset, the full pipeline should run comfortably in a typical local development environment.

Recommended targets:

```text
Data loading and transformation: within 1 minute
Database loading: within 1 minute
Single query response: within 3 seconds
Recommendation generation: within 10 seconds
```

### NFR-007. Testability

Core transformation and query functions shall be unit-testable.

Test targets include:

```text
Genre parsing
Actor parsing
Director extraction
Duplicate removal
Relationship table generation
Recommendation score calculation
```

# 3. Comparison System

## 3.1 Purpose

The comparison system explains and measures why `caracaldb` and `lynxes` are used for this project, compared with alternative database and DataFrame technologies.

The comparison must be a runnable benchmark, not only a design-level discussion. It should still be interpreted as a local toy-project benchmark rather than a production-grade benchmark.

## 3.2 Database Comparison

Comparison targets:

```text
caracaldb
SQLite
PostgreSQL
Neo4j
```

Comparison criteria:

```text
Installation and execution difficulty
Suitability for relational data storage
Suitability for graph-like relationship modeling
Query convenience
Suitability for local toy projects
Scalability
```

## 3.3 Database Comparison Table

| Criterion                   |    caracaldb |     SQLite | PostgreSQL |     Neo4j |
| --------------------------- | -----------: | ---------: | ---------: | --------: |
| Required by this project    |         High |        Low |        Low |       Low |
| Local setup convenience     |         High |  Very High |     Medium |    Medium |
| Relational table storage    |         High |       High |  Very High |    Medium |
| Graph relationship modeling |  Medium-High |     Medium |     Medium | Very High |
| SQL-style query suitability | High assumed |       High |  Very High |       Low |
| Graph traversal queries     |       Medium | Low-Medium |     Medium | Very High |
| Learning curve              |       Medium |        Low |     Medium |    Medium |
| Large-scale scalability     |       Medium |        Low |       High |      High |
| Toy project suitability     |         High |  Very High |     Medium |    Medium |
| Portfolio distinctiveness   |         High |        Low |     Medium |      High |

## 3.4 Database Evaluation

### caracaldb

`caracaldb` is the required database for this project. It serves as the main storage layer for entity and relationship tables.

Advantages:

```text
It directly satisfies the client-specified technology requirement.
It can store structured entities such as movies, people, genres, and keywords.
Relationship tables can represent knowledge graph edges.
```

Limitations:

```text
If it is not a graph-native database, complex multi-hop traversal may require additional application logic.
The implementation approach may depend on its supported query model.
```

Recommended use in this project:

```text
Store cleaned data
Manage entity and relationship tables
Support basic query processing
Retrieve data for analysis
```

### SQLite

SQLite is a simple local relational database alternative.

Advantages:

```text
It requires almost no setup.
It is easy to use with Python.
It is fast enough for small datasets.
```

Limitations:

```text
It has limited support for concurrency and large-scale workloads.
Graph traversal is difficult to express directly.
```

Reason for not selecting SQLite:

```text
The project requires caracaldb.
SQLite is less distinctive and does not satisfy the specified database requirement.
```

### PostgreSQL

PostgreSQL is a robust general-purpose relational database.

Advantages:

```text
It supports complex SQL queries.
It is suitable for larger datasets and indexing.
It supports JSON, arrays, and extensions.
```

Limitations:

```text
It may require too much setup for a toy project.
Graph traversal usually depends on joins or recursive queries.
```

Reason for not selecting PostgreSQL:

```text
It may be too heavy for the project scope.
caracaldb is the required database, so PostgreSQL is used only as a comparison target.
```

### Neo4j

Neo4j is a graph database alternative.

Advantages:

```text
It is highly suitable for node-edge modeling.
Multi-hop relationship traversal is natural.
It fits movie-actor-director-genre relationship analysis well.
```

Limitations:

```text
It requires learning a separate graph query language.
Its model differs from a table-based DataFrame pipeline.
It may be slightly heavy for a small toy project.
```

Reason for not selecting Neo4j:

```text
The project requires caracaldb.
Neo4j is suitable for a future graph-native extension.
```

## 3.5 DataFrame Library Comparison

Comparison targets:

```text
lynxes
pandas
Polars
Dask DataFrame
```

Comparison criteria:

```text
CSV processing convenience
Data cleaning convenience
Suitability for table transformation
Learning curve
Large-data handling
Database integration convenience
```

## 3.6 DataFrame Comparison Table

| Criterion                 |       lynxes |    pandas |    Polars | Dask DataFrame |
| ------------------------- | -----------: | --------: | --------: | -------------: |
| Required by this project  |         High |       Low |       Low |            Low |
| CSV loading               | High assumed | Very High | Very High |           High |
| Data cleaning             | High assumed | Very High |      High |         Medium |
| JSON-like column parsing  |  Medium-High |      High |    Medium |         Medium |
| Learning curve            |       Medium |       Low |    Medium |           High |
| Small-data suitability    |         High | Very High |      High |     Low-Medium |
| Large-data suitability    |       Medium |    Medium |      High |      Very High |
| Database integration      |  Medium-High | Very High |    Medium |         Medium |
| Toy project suitability   |         High | Very High |      High |            Low |
| Portfolio distinctiveness |         High |       Low |    Medium |         Medium |

## 3.7 DataFrame Library Evaluation

### lynxes

`lynxes` is the required DataFrame library for this project. It is used for CSV loading, data cleaning, entity extraction, and relationship table creation.

Advantages:

```text
It directly satisfies the client-specified DataFrame requirement.
It clearly demonstrates a DataFrame-based ETL structure.
It can be used to normalize nested movie data into entity and relationship tables.
```

Limitations:

```text
It may have fewer examples or ecosystem support than pandas.
Specific database integration may require custom implementation.
```

Recommended use in this project:

```text
CSV loading
Missing value handling
Duplicate removal
Entity table creation
Relationship table creation
Report table generation
```

### pandas

pandas is the most widely used Python DataFrame library.

Advantages:

```text
It has extensive documentation and examples.
It supports CSV, JSON, and SQL integration very well.
It is highly suitable for small datasets.
```

Limitations:

```text
It has limitations for very large datasets.
It is not the required library for this project.
```

Reason for not selecting pandas:

```text
The project requires lynxes.
pandas is used only as a comparison baseline.
```

### Polars

Polars is a high-performance DataFrame library.

Advantages:

```text
It is fast.
It handles larger datasets well.
It supports lazy evaluation.
```

Limitations:

```text
It has fewer learning resources than pandas.
Some Python ecosystem integrations may be less mature than pandas.
```

Reason for not selecting Polars:

```text
Although attractive for performance, the required library is lynxes.
For the TMDB 5000 dataset size, Polars is not necessary.
```

### Dask DataFrame

Dask DataFrame is designed for distributed or parallel DataFrame processing.

Advantages:

```text
It is suitable for large-scale data processing.
It provides a pandas-like interface.
It can use parallel computation.
```

Limitations:

```text
It is unnecessarily complex for this toy project.
Debugging and environment setup can be more difficult.
```

Reason for not selecting Dask:

```text
The dataset is not large enough to require Dask.
The added complexity is not justified.
The required DataFrame library is lynxes.
```

## 3.8 Final Technology Selection

The project uses:

```text
Database: caracaldb
DataFrame Library: lynxes
Programming Language: Python
```

Selection rationale:

```text
The selected stack satisfies the client-specified technology constraints.
The system clearly separates DataFrame-based ETL from database-based storage.
The knowledge graph can be represented through entity and relationship tables.
The implementation scope remains appropriate for a toy project.
The design can later be migrated or extended to Neo4j, PostgreSQL, pandas, or Polars.
```

## 3.9 Benchmark Workload

The comparison system must evaluate the 4x4 matrix using a knowledge graph-style workload. A single relational query such as "count Science Fiction movies" is not sufficient, because that favors traditional relational tools and does not reflect the purpose of this project.

The required benchmark workload is:

```text
Genre-based movie search
Actor-to-movie lookup
Director-to-movie lookup
Repeated director-actor collaboration analysis
Similar movie recommendation overlap analysis
```

The benchmark may repeat this workload multiple times to reduce noise and make DB/query-layer differences visible.

Recommended default:

```text
Benchmark target movie: Interstellar
Benchmark actor: Tom Hanks
Benchmark director: Christopher Nolan
Benchmark genre: Science Fiction
Workload repeats: 25
```

For each matrix cell, the benchmark should report:

```text
dataframe
database
status
total_seconds
dataframe_seconds
database_seconds
dataframe_loc
database_loc
movies
query_rows
notes
```

`query_rows` should represent the aggregate result count from all workload passes, not a single output table size.

## 3.10 Example Comparison System Output

The comparison system may be implemented as a separate module:

```text
src/compare_systems.py
```

Example output:

```text
# 4x4 Real Comparison Matrix

Each cell is:
status / total_seconds / dataframe_loc+database_loc / query_rows

| DataFrame \ Database | caracaldb | SQLite | PostgreSQL | Neo4j |
| -------------------- | --------: | -----: | ---------: | ----: |
| lynxes               | ok        | ok     | skipped/ok | skipped/ok |
| pandas               | ok        | ok     | skipped/ok | skipped/ok |
| Polars               | ok        | ok     | skipped/ok | skipped/ok |
| Dask DataFrame       | ok        | ok     | skipped/ok | skipped/ok |
```

The system should also export a detailed CSV file containing every matrix cell.

PostgreSQL and Neo4j are external services. If connection settings are not available, their cells should be marked as `skipped` with a clear note. If connection settings are available, the benchmark should execute them.

## 3.11 Evaluation Criteria

From a client perspective, the project can be evaluated using the following criteria:

```text
Whether caracaldb and lynxes are used as required
Whether the raw CSV data is loaded and cleaned correctly
Whether entities and relationships are clearly separated
Whether the data is stored in a structure suitable for knowledge graph-style queries
Whether collaboration analysis, genre search, and similar movie recommendation work correctly
Whether the comparison system runs a real DataFrame x Database matrix
Whether the benchmark uses knowledge graph-style workload rather than a single relational lookup
Whether timing, code length, row count, status, and notes are exported
Whether the README and execution instructions are clear
Whether tests or validation results are included
```


