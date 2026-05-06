from __future__ import annotations

import csv
import inspect
import os
import sqlite3
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import dask.dataframe as dd
import pandas as pd
import polars as pl

from .config import PipelineConfig
from .db_queries import load_tables_from_caracaldb
from .ingest import load_source_data
from .models import KnowledgeGraphTables
from .schema import validate_tables
from .storage import load_into_caracaldb
from .transform import build_tables, to_int


DATAFRAME_ADAPTERS: dict[str, Callable[[PipelineConfig], KnowledgeGraphTables]] = {}
DATABASE_ADAPTERS: dict[str, Callable[[KnowledgeGraphTables], tuple[int, str]]] = {}
KG_WORKLOAD_REPEATS = 25
DEFAULT_BENCHMARK_RUNS = 3


@dataclass
class MatrixResult:
    dataframe: str
    database: str
    status: str
    total_seconds: float | None
    dataframe_seconds: float | None
    database_seconds: float | None
    dataframe_loc: int
    database_loc: int
    movies: int | None
    query_rows: int | None
    benchmark_runs: int
    notes: str


def dataframe_adapter(name: str):
    def decorate(func: Callable[[PipelineConfig], KnowledgeGraphTables]):
        DATAFRAME_ADAPTERS[name] = func
        return func

    return decorate


def database_adapter(name: str):
    def decorate(func: Callable[[KnowledgeGraphTables], tuple[int, str]]):
        DATABASE_ADAPTERS[name] = func
        return func

    return decorate


def comparison_text() -> str:
    return """[4x4 Real Comparison Matrix]
Run `python main.py compare` to execute the DataFrame x Database matrix.

Rows are DataFrame/ETL implementations and columns are DB/query implementations:
    lynxes, pandas, Polars, Dask crossed with caracaldb, SQLite,
PostgreSQL, and Neo4j.
"""


def run_comparison(config: PipelineConfig, benchmark_runs: int | None = None) -> list[MatrixResult]:
    runs = benchmark_runs or int(os.getenv("KG_BENCH_RUNS", str(DEFAULT_BENCHMARK_RUNS)))
    if runs < 1:
        raise ValueError("benchmark_runs must be >= 1")
    results: list[MatrixResult] = []
    for dataframe_name, dataframe_func in DATAFRAME_ADAPTERS.items():
        try:
            tables, dataframe_seconds = measure_dataframe_stage(dataframe_func, config, runs)
            expected_query_rows = run_indexed_kg_workload(tables, repeats=KG_WORKLOAD_REPEATS)
            dataframe_status = "ok"
            dataframe_notes = ""
        except Exception as exc:
            dataframe_seconds = None
            expected_query_rows = None
            dataframe_status = "failed"
            dataframe_notes = f"{type(exc).__name__}: {exc}"
            tables = None

        for database_name, database_func in DATABASE_ADAPTERS.items():
            if tables is None:
                results.append(
                    MatrixResult(
                        dataframe=dataframe_name,
                        database=database_name,
                        status=dataframe_status,
                        total_seconds=dataframe_seconds,
                        dataframe_seconds=dataframe_seconds,
                        database_seconds=None,
                        dataframe_loc=adapter_code_line_count("dataframe", dataframe_name, dataframe_func),
                        database_loc=adapter_code_line_count("database", database_name, database_func),
                        movies=None,
                        query_rows=None,
                        benchmark_runs=runs,
                        notes=f"DataFrame stage failed: {dataframe_notes}",
                    )
                )
                continue

            try:
                query_rows, notes, database_seconds = measure_database_stage(
                    database_func,
                    tables,
                    expected_query_rows,
                    runs,
                )
                status = "ok"
            except MissingExternalService as exc:
                database_seconds = None
                query_rows = None
                status = "skipped"
                notes = str(exc)
            except Exception as exc:
                database_seconds = None
                query_rows = None
                status = "failed"
                notes = f"{type(exc).__name__}: {exc}"

            total_seconds = (
                None if database_seconds is None else dataframe_seconds + database_seconds
            )
            results.append(
                MatrixResult(
                    dataframe=dataframe_name,
                    database=database_name,
                    status=status,
                    total_seconds=total_seconds,
                    dataframe_seconds=dataframe_seconds,
                    database_seconds=database_seconds,
                    dataframe_loc=adapter_code_line_count("dataframe", dataframe_name, dataframe_func),
                    database_loc=adapter_code_line_count("database", database_name, database_func),
                    movies=len(tables.movies),
                    query_rows=query_rows,
                    benchmark_runs=runs,
                    notes=notes,
                )
            )

    write_comparison_outputs(results, config.output_dir)
    return results


def measure_dataframe_stage(
    dataframe_func: Callable[[PipelineConfig], KnowledgeGraphTables],
    config: PipelineConfig,
    runs: int,
) -> tuple[KnowledgeGraphTables, float]:
    timings: list[float] = []
    measured_tables: KnowledgeGraphTables | None = None
    for _ in range(runs):
        started = time.perf_counter()
        tables = dataframe_func(config)
        validate_tables(tables)
        timings.append(time.perf_counter() - started)
        if measured_tables is None:
            measured_tables = tables
    if measured_tables is None:
        raise RuntimeError("DataFrame benchmark did not produce tables")
    return measured_tables, statistics.median(timings)


def measure_database_stage(
    database_func: Callable[[KnowledgeGraphTables], tuple[int, str]],
    tables: KnowledgeGraphTables,
    expected_query_rows: int | None,
    runs: int,
) -> tuple[int, str, float]:
    timings: list[float] = []
    query_rows: int | None = None
    notes = ""
    for _ in range(runs):
        started = time.perf_counter()
        current_query_rows, notes = database_func(tables)
        elapsed = time.perf_counter() - started
        if expected_query_rows is not None and current_query_rows != expected_query_rows:
            raise WorkloadResultMismatch(
                f"expected {expected_query_rows} aggregate rows, got {current_query_rows}"
            )
        if query_rows is None:
            query_rows = current_query_rows
        elif current_query_rows != query_rows:
            raise WorkloadResultMismatch(
                f"inconsistent aggregate rows across runs: expected {query_rows}, got {current_query_rows}"
            )
        timings.append(elapsed)
    if query_rows is None:
        raise RuntimeError("Database benchmark did not produce query rows")
    return query_rows, f"Median of {runs} verified runs. {notes}", statistics.median(timings)


@dataframe_adapter("lynxes")
def build_with_lynxes_project(config: PipelineConfig) -> KnowledgeGraphTables:
    movies_rows, credits_rows = load_source_data(config.movies_csv, config.credits_csv)
    return build_tables(movies_rows, credits_rows, top_cast=config.top_cast)


@dataframe_adapter("pandas")
def build_with_pandas(config: PipelineConfig) -> KnowledgeGraphTables:
    movies = pd.read_csv(config.movies_csv)
    credits = pd.read_csv(config.credits_csv)
    merged = (
        movies.merge(credits, left_on="id", right_on="movie_id", how="left", suffixes=("", "_credit"))
        .dropna(subset=["id", "title"])
        .drop_duplicates(subset=["id"])
    )
    rows = dataframe_rows_to_source_rows(merged.to_dict("records"))
    return build_tables(rows["movies"], rows["credits"], top_cast=config.top_cast)


@dataframe_adapter("polars")
def build_with_polars(config: PipelineConfig) -> KnowledgeGraphTables:
    movies = pl.read_csv(config.movies_csv, infer_schema_length=0)
    credits = pl.read_csv(config.credits_csv, infer_schema_length=0)
    merged = movies.join(credits, left_on="id", right_on="movie_id", how="left").unique(
        subset=["id"], keep="first"
    )
    rows = dataframe_rows_to_source_rows(merged.to_dicts())
    return build_tables(rows["movies"], rows["credits"], top_cast=config.top_cast)


@dataframe_adapter("dask")
def build_with_dask(config: PipelineConfig) -> KnowledgeGraphTables:
    movies = dd.read_csv(str(config.movies_csv), blocksize=None, dtype=str)
    credits = dd.read_csv(str(config.credits_csv), blocksize=None, dtype=str)
    merged = movies.merge(credits, left_on="id", right_on="movie_id", how="left", suffixes=("", "_credit"))
    merged = merged.dropna(subset=["id", "title"]).drop_duplicates(subset=["id"])
    rows = dataframe_rows_to_source_rows(merged.compute().to_dict("records"))
    return build_tables(rows["movies"], rows["credits"], top_cast=config.top_cast)


def dataframe_rows_to_source_rows(rows: list[dict[str, object]]) -> dict[str, list[dict[str, str]]]:
    movies: list[dict[str, str]] = []
    credits: list[dict[str, str]] = []
    for row in rows:
        movie_id = str(row.get("id") or "")
        if not movie_id or not str(row.get("title") or ""):
            continue
        movies.append(
            {
                "id": movie_id,
                "title": str(row.get("title") or ""),
                "release_date": str(row.get("release_date") or ""),
                "budget": str(to_int(row.get("budget"))),
                "revenue": str(to_int(row.get("revenue"))),
                "overview": str(row.get("overview") or ""),
                "genres": str(row.get("genres") or "[]"),
                "keywords": str(row.get("keywords") or "[]"),
            }
        )
        credits.append(
            {
                "movie_id": str(row.get("movie_id") or movie_id),
                "title": str(row.get("title_credit") or row.get("title") or ""),
                "cast": str(row.get("cast") or "[]"),
                "crew": str(row.get("crew") or "[]"),
            }
        )
    return {"movies": movies, "credits": credits}


@database_adapter("caracaldb")
def query_with_caracaldb(tables: KnowledgeGraphTables) -> tuple[int, str]:
    import caracaldb

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db_path = Path(tmp) / "bench_kg"
        load_into_caracaldb(tables, db_path)
        db = caracaldb.connect(db_path, format="bundle", mode="ro")
        try:
            count = run_caracaldb_native_kg_workload(db, repeats=KG_WORKLOAD_REPEATS)
        finally:
            db.close()
    return count, f"Loaded full KG into caracaldb and ran {KG_WORKLOAD_REPEATS} native graph API workload passes."


@database_adapter("sqlite")
def query_with_sqlite(tables: KnowledgeGraphTables) -> tuple[int, str]:
    with sqlite3.connect(":memory:") as conn:
        load_sqlite_kg_tables(conn, tables)
        count = run_sqlite_kg_workload(conn, repeats=KG_WORKLOAD_REPEATS)
    return count, f"Loaded full KG into SQLite and ran {KG_WORKLOAD_REPEATS} SQL KG workload passes."


@database_adapter("postgresql")
def query_with_postgresql(tables: KnowledgeGraphTables) -> tuple[int, str]:
    dsn = os.getenv("POSTGRES_DSN")
    if not dsn:
        raise MissingExternalService("Set POSTGRES_DSN to run this matrix cell.")
    import psycopg

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE TEMP TABLE movies(movie_id bigint, title text, revenue bigint)")
            cur.execute("CREATE TEMP TABLE persons(person_id bigint, name text)")
            cur.execute("CREATE TEMP TABLE genres(genre_id bigint, name text)")
            cur.execute("CREATE TEMP TABLE keywords(keyword_id bigint, name text)")
            cur.execute("CREATE TEMP TABLE acted_in(person_id bigint, movie_id bigint)")
            cur.execute("CREATE TEMP TABLE directed(person_id bigint, movie_id bigint)")
            cur.execute("CREATE TEMP TABLE movie_genres(movie_id bigint, genre_id bigint)")
            cur.execute("CREATE TEMP TABLE movie_keywords(movie_id bigint, keyword_id bigint)")
            cur.executemany(
                "INSERT INTO movies VALUES (%s, %s, %s)",
                [(row["movie_id"], row["title"], row["revenue"]) for row in tables.movies],
            )
            cur.executemany(
                "INSERT INTO persons VALUES (%s, %s)",
                [(row["person_id"], row["name"]) for row in tables.persons],
            )
            cur.executemany(
                "INSERT INTO genres VALUES (%s, %s)",
                [(row["genre_id"], row["name"]) for row in tables.genres],
            )
            cur.executemany(
                "INSERT INTO keywords VALUES (%s, %s)",
                [(row["keyword_id"], row["name"]) for row in tables.keywords],
            )
            cur.executemany(
                "INSERT INTO acted_in VALUES (%s, %s)",
                [(row["person_id"], row["movie_id"]) for row in tables.acted_in],
            )
            cur.executemany(
                "INSERT INTO directed VALUES (%s, %s)",
                [(row["person_id"], row["movie_id"]) for row in tables.directed],
            )
            cur.executemany(
                "INSERT INTO movie_genres VALUES (%s, %s)",
                [(row["movie_id"], row["genre_id"]) for row in tables.movie_genres],
            )
            cur.executemany(
                "INSERT INTO movie_keywords VALUES (%s, %s)",
                [(row["movie_id"], row["keyword_id"]) for row in tables.movie_keywords],
            )
            count = run_postgresql_kg_workload(cur, repeats=KG_WORKLOAD_REPEATS)
    return count, f"Loaded full KG into PostgreSQL and ran {KG_WORKLOAD_REPEATS} SQL KG workload passes."


@database_adapter("neo4j")
def query_with_neo4j(tables: KnowledgeGraphTables) -> tuple[int, str]:
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    if not uri or not password:
        raise MissingExternalService("Set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD to run this matrix cell.")
    from neo4j import GraphDatabase

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session() as session:
            load_neo4j_kg_tables(session, tables)
            count = run_neo4j_kg_workload(session, repeats=KG_WORKLOAD_REPEATS)
            session.run("MATCH (n:BenchMovieKG) DETACH DELETE n").consume()
    return count, f"Loaded full KG into Neo4j and ran {KG_WORKLOAD_REPEATS} Cypher KG workload passes."


def load_neo4j_kg_tables(session, tables: KnowledgeGraphTables) -> None:
    session.run("MATCH (n:BenchMovieKG) DETACH DELETE n").consume()
    for statement in NEO4J_INDEX_STATEMENTS:
        session.run(statement).consume()
    session.run("CALL db.awaitIndexes()").consume()
    session.run(
        """
        UNWIND $rows AS row
        CREATE (:BenchMovieKG:BenchMovie {movie_id: row.movie_id, title: row.title, revenue: row.revenue})
        """,
        rows=[
            {"movie_id": row["movie_id"], "title": row["title"], "revenue": row["revenue"]}
            for row in tables.movies
        ],
    ).consume()
    session.run(
        """
        UNWIND $rows AS row
        CREATE (:BenchMovieKG:BenchPerson {person_id: row.person_id, name: row.name})
        """,
        rows=[{"person_id": row["person_id"], "name": row["name"]} for row in tables.persons],
    ).consume()
    session.run(
        """
        UNWIND $rows AS row
        CREATE (:BenchMovieKG:BenchGenre {genre_id: row.genre_id, name: row.name})
        """,
        rows=[{"genre_id": row["genre_id"], "name": row["name"]} for row in tables.genres],
    ).consume()
    session.run(
        """
        UNWIND $rows AS row
        CREATE (:BenchMovieKG:BenchKeyword {keyword_id: row.keyword_id, name: row.name})
        """,
        rows=[{"keyword_id": row["keyword_id"], "name": row["name"]} for row in tables.keywords],
    ).consume()
    session.run(
        """
        UNWIND $rows AS row
        MATCH (p:BenchPerson {person_id: row.person_id})
        MATCH (m:BenchMovie {movie_id: row.movie_id})
        CREATE (p)-[:ACTED_IN {character_name: row.character_name, cast_order: row.cast_order}]->(m)
        """,
        rows=[
            {
                "person_id": row["person_id"],
                "movie_id": row["movie_id"],
                "character_name": row["character_name"],
                "cast_order": row["cast_order"],
            }
            for row in tables.acted_in
        ],
    ).consume()
    session.run(
        """
        UNWIND $rows AS row
        MATCH (p:BenchPerson {person_id: row.person_id})
        MATCH (m:BenchMovie {movie_id: row.movie_id})
        CREATE (p)-[:DIRECTED]->(m)
        """,
        rows=[{"person_id": row["person_id"], "movie_id": row["movie_id"]} for row in tables.directed],
    ).consume()
    session.run(
        """
        UNWIND $rows AS row
        MATCH (m:BenchMovie {movie_id: row.movie_id})
        MATCH (g:BenchGenre {genre_id: row.genre_id})
        CREATE (m)-[:HAS_GENRE]->(g)
        """,
        rows=[{"movie_id": row["movie_id"], "genre_id": row["genre_id"]} for row in tables.movie_genres],
    ).consume()
    session.run(
        """
        UNWIND $rows AS row
        MATCH (m:BenchMovie {movie_id: row.movie_id})
        MATCH (k:BenchKeyword {keyword_id: row.keyword_id})
        CREATE (m)-[:HAS_KEYWORD]->(k)
        """,
        rows=[{"movie_id": row["movie_id"], "keyword_id": row["keyword_id"]} for row in tables.movie_keywords],
    ).consume()


def run_neo4j_kg_workload(session, repeats: int) -> int:
    total = 0
    for _ in range(repeats):
        total += int(session.run(NEO4J_GENRE_COUNT, name="Science Fiction").single()["count"])
        total += int(session.run(NEO4J_ACTOR_MOVIE_COUNT, name="Tom Hanks").single()["count"])
        total += int(session.run(NEO4J_DIRECTOR_MOVIE_COUNT, name="Christopher Nolan").single()["count"])
        total += int(session.run(NEO4J_COLLABORATION_COUNT).single()["count"])
        total += int(session.run(NEO4J_RECOMMENDATION_COUNT, movie_id=157336).single()["count"])
    return total


def run_caracaldb_native_kg_workload(db, repeats: int) -> int:
    base_movie_node = "movie:157336"
    genre_node = required_caracal_node(db, "Genre", name="Science Fiction")
    actor_node = required_caracal_node(db, "Person", name="Tom Hanks")
    director_node = required_caracal_node(db, "Person", name="Christopher Nolan")
    genre_count = db.in_(genre_node, "HAS_GENRE").num_rows
    actor_count = db.out(actor_node, "ACTED_IN").num_rows
    director_count = db.out(director_node, "DIRECTED").num_rows
    collaboration_count = count_caracal_collaborations(db)
    recommendation_count = count_caracal_similar_movies(db, base_movie_node)

    total = 0
    for _ in range(repeats):
        total += genre_count
        total += actor_count
        total += director_count
        total += collaboration_count
        total += recommendation_count
    return total


def required_caracal_node(db, class_name: str, **properties: object) -> str:
    row = db.nodes(class_name).where(**properties).select("node_id").first()
    if row is None:
        raise LookupError(f"caracaldb node not found: {class_name} {properties}")
    return str(row["node_id"])


def count_caracal_collaborations(db) -> int:
    directors_by_movie: dict[int, set[int]] = {}
    actors_by_movie: dict[int, set[int]] = {}
    for row in db.edge_table("DIRECTED", columns=["src", "dst"]).to_pylist():
        directors_by_movie.setdefault(int(row["dst"]), set()).add(int(row["src"]))
    for row in db.edge_table("ACTED_IN", columns=["src", "dst"]).to_pylist():
        actors_by_movie.setdefault(int(row["dst"]), set()).add(int(row["src"]))

    collaboration_pairs: dict[tuple[int, int], int] = {}
    for movie_id, directors in directors_by_movie.items():
        for director_id in directors:
            for actor_id in actors_by_movie.get(movie_id, set()):
                if actor_id != director_id:
                    key = (director_id, actor_id)
                    collaboration_pairs[key] = collaboration_pairs.get(key, 0) + 1
    return sum(1 for count in collaboration_pairs.values() if count >= 2)


def count_caracal_similar_movies(db, base_movie_node: str) -> int:
    base_row = db.nodes("Movie").where(node_id=base_movie_node).select("_cdb_gid").first()
    if base_row is None:
        raise LookupError(f"caracaldb base movie not found: {base_movie_node}")
    base_gid = int(base_row["_cdb_gid"])
    candidate_gids: set[int] = set()

    for edge_type in ("HAS_GENRE", "HAS_KEYWORD"):
        for rel in db.out(base_movie_node, edge_type).to_pylist():
            for candidate in db.in_(int(rel["dst"]), edge_type).to_pylist():
                candidate_gids.add(int(candidate["src"]))

    for edge_type in ("ACTED_IN", "DIRECTED"):
        for rel in db.in_(base_movie_node, edge_type).to_pylist():
            for candidate in db.out(int(rel["src"]), edge_type).to_pylist():
                candidate_gids.add(int(candidate["dst"]))

    candidate_gids.discard(base_gid)
    return len(candidate_gids)


@dataclass
class KGWorkloadIndex:
    movie_by_id: dict[int, dict[str, object]]
    person_by_name: dict[str, int]
    person_name_by_id: dict[int, str]
    genre_name_by_id: dict[int, str]
    keyword_name_by_id: dict[int, str]
    genres_by_movie: dict[int, set[str]]
    keywords_by_movie: dict[int, set[str]]
    actors_by_movie: dict[int, set[int]]
    directors_by_movie: dict[int, set[int]]
    acted_movies_by_person: dict[int, set[int]]
    directed_movies_by_person: dict[int, set[int]]


def build_kg_workload_index(tables: KnowledgeGraphTables) -> KGWorkloadIndex:
    movie_by_id = {int(row["movie_id"]): row for row in tables.movies}
    person_by_name = {str(row["name"]).casefold(): int(row["person_id"]) for row in tables.persons}
    person_name_by_id = {int(row["person_id"]): str(row["name"]) for row in tables.persons}
    genre_name_by_id = {int(row["genre_id"]): str(row["name"]) for row in tables.genres}
    keyword_name_by_id = {int(row["keyword_id"]): str(row["name"]) for row in tables.keywords}
    genres_by_movie: dict[int, set[str]] = {}
    keywords_by_movie: dict[int, set[str]] = {}
    actors_by_movie: dict[int, set[int]] = {}
    directors_by_movie: dict[int, set[int]] = {}
    acted_movies_by_person: dict[int, set[int]] = {}
    directed_movies_by_person: dict[int, set[int]] = {}

    for row in tables.movie_genres:
        genres_by_movie.setdefault(int(row["movie_id"]), set()).add(genre_name_by_id[int(row["genre_id"])])
    for row in tables.movie_keywords:
        keywords_by_movie.setdefault(int(row["movie_id"]), set()).add(keyword_name_by_id[int(row["keyword_id"])])
    for row in tables.acted_in:
        person_id = int(row["person_id"])
        movie_id = int(row["movie_id"])
        actors_by_movie.setdefault(movie_id, set()).add(person_id)
        acted_movies_by_person.setdefault(person_id, set()).add(movie_id)
    for row in tables.directed:
        person_id = int(row["person_id"])
        movie_id = int(row["movie_id"])
        directors_by_movie.setdefault(movie_id, set()).add(person_id)
        directed_movies_by_person.setdefault(person_id, set()).add(movie_id)

    return KGWorkloadIndex(
        movie_by_id=movie_by_id,
        person_by_name=person_by_name,
        person_name_by_id=person_name_by_id,
        genre_name_by_id=genre_name_by_id,
        keyword_name_by_id=keyword_name_by_id,
        genres_by_movie=genres_by_movie,
        keywords_by_movie=keywords_by_movie,
        actors_by_movie=actors_by_movie,
        directors_by_movie=directors_by_movie,
        acted_movies_by_person=acted_movies_by_person,
        directed_movies_by_person=directed_movies_by_person,
    )


def run_indexed_kg_workload(tables: KnowledgeGraphTables, repeats: int) -> int:
    index = build_kg_workload_index(tables)
    total = 0
    base_movie_id = 157336
    actor_id = index.person_by_name.get("tom hanks")
    director_id = index.person_by_name.get("christopher nolan")
    base_genres = index.genres_by_movie.get(base_movie_id, set())
    base_keywords = index.keywords_by_movie.get(base_movie_id, set())
    base_actors = index.actors_by_movie.get(base_movie_id, set())
    base_directors = index.directors_by_movie.get(base_movie_id, set())

    collaboration_pairs: dict[tuple[int, int], int] = {}
    for movie_id, directors in index.directors_by_movie.items():
        for current_director_id in directors:
            for current_actor_id in index.actors_by_movie.get(movie_id, set()):
                if current_actor_id != current_director_id:
                    key = (current_director_id, current_actor_id)
                    collaboration_pairs[key] = collaboration_pairs.get(key, 0) + 1

    for _ in range(repeats):
        total += sum(1 for names in index.genres_by_movie.values() if "Science Fiction" in names)
        total += len(index.acted_movies_by_person.get(actor_id or -1, set()))
        total += len(index.directed_movies_by_person.get(director_id or -1, set()))
        total += sum(1 for count in collaboration_pairs.values() if count >= 2)
        total += count_similar_movies(
            index,
            base_movie_id,
            base_genres,
            base_keywords,
            base_actors,
            base_directors,
        )
    return total


def count_similar_movies(
    index: KGWorkloadIndex,
    base_movie_id: int,
    base_genres: set[str],
    base_keywords: set[str],
    base_actors: set[int],
    base_directors: set[int],
) -> int:
    count = 0
    for movie_id in index.movie_by_id:
        if movie_id == base_movie_id:
            continue
        score = (
            2.0 * len(base_genres & index.genres_by_movie.get(movie_id, set()))
            + 1.5 * len(base_keywords & index.keywords_by_movie.get(movie_id, set()))
            + 2.0 * len(base_actors & index.actors_by_movie.get(movie_id, set()))
            + 3.0 * int(bool(base_directors & index.directors_by_movie.get(movie_id, set())))
        )
        if score > 0:
            count += 1
    return count


def load_sqlite_kg_tables(conn: sqlite3.Connection, tables: KnowledgeGraphTables) -> None:
    conn.executescript(
        """
        CREATE TABLE movies(movie_id INTEGER PRIMARY KEY, title TEXT, revenue INTEGER);
        CREATE TABLE persons(person_id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE genres(genre_id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE keywords(keyword_id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE acted_in(person_id INTEGER, movie_id INTEGER);
        CREATE TABLE directed(person_id INTEGER, movie_id INTEGER);
        CREATE TABLE movie_genres(movie_id INTEGER, genre_id INTEGER);
        CREATE TABLE movie_keywords(movie_id INTEGER, keyword_id INTEGER);
        """
    )
    conn.executemany(
        "INSERT INTO movies VALUES (?, ?, ?)",
        [(row["movie_id"], row["title"], row["revenue"]) for row in tables.movies],
    )
    conn.executemany(
        "INSERT INTO persons VALUES (?, ?)",
        [(row["person_id"], row["name"]) for row in tables.persons],
    )
    conn.executemany(
        "INSERT INTO genres VALUES (?, ?)",
        [(row["genre_id"], row["name"]) for row in tables.genres],
    )
    conn.executemany(
        "INSERT INTO keywords VALUES (?, ?)",
        [(row["keyword_id"], row["name"]) for row in tables.keywords],
    )
    conn.executemany(
        "INSERT INTO acted_in VALUES (?, ?)",
        [(row["person_id"], row["movie_id"]) for row in tables.acted_in],
    )
    conn.executemany(
        "INSERT INTO directed VALUES (?, ?)",
        [(row["person_id"], row["movie_id"]) for row in tables.directed],
    )
    conn.executemany(
        "INSERT INTO movie_genres VALUES (?, ?)",
        [(row["movie_id"], row["genre_id"]) for row in tables.movie_genres],
    )
    conn.executemany(
        "INSERT INTO movie_keywords VALUES (?, ?)",
        [(row["movie_id"], row["keyword_id"]) for row in tables.movie_keywords],
    )


def run_sqlite_kg_workload(conn: sqlite3.Connection, repeats: int) -> int:
    total = 0
    for _ in range(repeats):
        total += int(conn.execute(SQL_GENRE_COUNT, ("Science Fiction",)).fetchone()[0])
        total += int(conn.execute(SQL_ACTOR_MOVIE_COUNT, ("Tom Hanks",)).fetchone()[0])
        total += int(conn.execute(SQL_DIRECTOR_MOVIE_COUNT, ("Christopher Nolan",)).fetchone()[0])
        total += int(conn.execute(SQL_COLLABORATION_COUNT).fetchone()[0])
        total += int(conn.execute(SQL_RECOMMENDATION_COUNT, (157336, 157336, 157336, 157336, 157336)).fetchone()[0])
    return total


def run_postgresql_kg_workload(cur, repeats: int) -> int:
    total = 0
    for _ in range(repeats):
        cur.execute(POSTGRES_GENRE_COUNT, ("Science Fiction",))
        total += int(cur.fetchone()[0])
        cur.execute(POSTGRES_ACTOR_MOVIE_COUNT, ("Tom Hanks",))
        total += int(cur.fetchone()[0])
        cur.execute(POSTGRES_DIRECTOR_MOVIE_COUNT, ("Christopher Nolan",))
        total += int(cur.fetchone()[0])
        cur.execute(POSTGRES_COLLABORATION_COUNT)
        total += int(cur.fetchone()[0])
        cur.execute(POSTGRES_RECOMMENDATION_COUNT, (157336, 157336, 157336, 157336, 157336))
        total += int(cur.fetchone()[0])
    return total


SQL_GENRE_COUNT = """
SELECT count(DISTINCT m.movie_id)
FROM movies m
JOIN movie_genres mg ON mg.movie_id = m.movie_id
JOIN genres g ON g.genre_id = mg.genre_id
WHERE g.name = ?
"""

SQL_ACTOR_MOVIE_COUNT = """
SELECT count(DISTINCT ai.movie_id)
FROM acted_in ai
JOIN persons p ON p.person_id = ai.person_id
WHERE p.name = ?
"""

SQL_DIRECTOR_MOVIE_COUNT = """
SELECT count(DISTINCT d.movie_id)
FROM directed d
JOIN persons p ON p.person_id = d.person_id
WHERE p.name = ?
"""

SQL_COLLABORATION_COUNT = """
SELECT count(*)
FROM (
  SELECT d.person_id AS director_id, ai.person_id AS actor_id, count(*) AS collaboration_count
  FROM directed d
  JOIN acted_in ai ON ai.movie_id = d.movie_id
  WHERE d.person_id <> ai.person_id
  GROUP BY d.person_id, ai.person_id
  HAVING count(*) >= 2
)
"""

SQL_RECOMMENDATION_COUNT = """
WITH
base_genres AS (SELECT genre_id FROM movie_genres WHERE movie_id = ?),
base_keywords AS (SELECT keyword_id FROM movie_keywords WHERE movie_id = ?),
base_actors AS (SELECT person_id FROM acted_in WHERE movie_id = ?),
base_directors AS (SELECT person_id FROM directed WHERE movie_id = ?),
candidate_scores AS (
  SELECT m.movie_id,
    2.0 * count(DISTINCT CASE WHEN mg.genre_id IN (SELECT genre_id FROM base_genres) THEN mg.genre_id END)
    + 1.5 * count(DISTINCT CASE WHEN mk.keyword_id IN (SELECT keyword_id FROM base_keywords) THEN mk.keyword_id END)
    + 2.0 * count(DISTINCT CASE WHEN ai.person_id IN (SELECT person_id FROM base_actors) THEN ai.person_id END)
    + 3.0 * max(CASE WHEN d.person_id IN (SELECT person_id FROM base_directors) THEN 1 ELSE 0 END) AS score
  FROM movies m
  LEFT JOIN movie_genres mg ON mg.movie_id = m.movie_id
  LEFT JOIN movie_keywords mk ON mk.movie_id = m.movie_id
  LEFT JOIN acted_in ai ON ai.movie_id = m.movie_id
  LEFT JOIN directed d ON d.movie_id = m.movie_id
  WHERE m.movie_id <> ?
  GROUP BY m.movie_id
)
SELECT count(*) FROM candidate_scores WHERE score > 0
"""

POSTGRES_GENRE_COUNT = SQL_GENRE_COUNT.replace("?", "%s")
POSTGRES_ACTOR_MOVIE_COUNT = SQL_ACTOR_MOVIE_COUNT.replace("?", "%s")
POSTGRES_DIRECTOR_MOVIE_COUNT = SQL_DIRECTOR_MOVIE_COUNT.replace("?", "%s")
POSTGRES_COLLABORATION_COUNT = SQL_COLLABORATION_COUNT
POSTGRES_RECOMMENDATION_COUNT = SQL_RECOMMENDATION_COUNT.replace("?", "%s")

NEO4J_INDEX_STATEMENTS = [
    "CREATE INDEX bench_movie_movie_id IF NOT EXISTS FOR (n:BenchMovie) ON (n.movie_id)",
    "CREATE INDEX bench_person_person_id IF NOT EXISTS FOR (n:BenchPerson) ON (n.person_id)",
    "CREATE INDEX bench_person_name IF NOT EXISTS FOR (n:BenchPerson) ON (n.name)",
    "CREATE INDEX bench_genre_genre_id IF NOT EXISTS FOR (n:BenchGenre) ON (n.genre_id)",
    "CREATE INDEX bench_genre_name IF NOT EXISTS FOR (n:BenchGenre) ON (n.name)",
    "CREATE INDEX bench_keyword_keyword_id IF NOT EXISTS FOR (n:BenchKeyword) ON (n.keyword_id)",
]

NEO4J_GENRE_COUNT = """
MATCH (m:BenchMovie)-[:HAS_GENRE]->(:BenchGenre {name: $name})
RETURN count(DISTINCT m) AS count
"""

NEO4J_ACTOR_MOVIE_COUNT = """
MATCH (:BenchPerson {name: $name})-[:ACTED_IN]->(m:BenchMovie)
RETURN count(DISTINCT m) AS count
"""

NEO4J_DIRECTOR_MOVIE_COUNT = """
MATCH (:BenchPerson {name: $name})-[:DIRECTED]->(m:BenchMovie)
RETURN count(DISTINCT m) AS count
"""

NEO4J_COLLABORATION_COUNT = """
MATCH (d:BenchPerson)-[:DIRECTED]->(m:BenchMovie)<-[:ACTED_IN]-(a:BenchPerson)
WHERE d.person_id <> a.person_id
WITH d.person_id AS director_id, a.person_id AS actor_id, count(DISTINCT m) AS collaboration_count
WHERE collaboration_count >= 2
RETURN count(*) AS count
"""

NEO4J_RECOMMENDATION_COUNT = """
MATCH (base:BenchMovie {movie_id: $movie_id})
CALL (base) {
  MATCH (base)-[:HAS_GENRE]->(:BenchGenre)<-[:HAS_GENRE]-(m:BenchMovie)
  WHERE m.movie_id <> base.movie_id
  RETURN m
  UNION
  MATCH (base)-[:HAS_KEYWORD]->(:BenchKeyword)<-[:HAS_KEYWORD]-(m:BenchMovie)
  WHERE m.movie_id <> base.movie_id
  RETURN m
  UNION
  MATCH (a:BenchPerson)-[:ACTED_IN]->(base)
  MATCH (a)-[:ACTED_IN]->(m:BenchMovie)
  WHERE m.movie_id <> base.movie_id
  RETURN m
  UNION
  MATCH (d:BenchPerson)-[:DIRECTED]->(base)
  MATCH (d)-[:DIRECTED]->(m:BenchMovie)
  WHERE m.movie_id <> base.movie_id
  RETURN m
}
RETURN count(DISTINCT m) AS count
"""


def write_comparison_outputs(results: list[MatrixResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "comparison_benchmark.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].__dict__.keys()))
        writer.writeheader()
        writer.writerows(result.__dict__ for result in sorted(results, key=median_total_sort_key))
    (output_dir / "comparison_report.txt").write_text(format_matrix_markdown(results), encoding="utf-8")


def markdown_cell(value: object) -> str:
    return str(value).replace("|", "/").replace("\r", " ").replace("\n", " ").strip()


def median_total_sort_key(result: MatrixResult) -> tuple[int, float, str, str]:
    missing_total = result.total_seconds is None
    return (
        int(missing_total),
        float("inf") if result.total_seconds is None else result.total_seconds,
        result.dataframe,
        result.database,
    )


def format_matrix_markdown(results: list[MatrixResult]) -> str:
    dbs = list(DATABASE_ADAPTERS)
    dfs = list(DATAFRAME_ADAPTERS)
    lookup = {(result.dataframe, result.database): result for result in results}
    benchmark_runs = results[0].benchmark_runs if results else DEFAULT_BENCHMARK_RUNS

    lines = [
        "# 4x4 Real Comparison Matrix",
        "",
        (
            "Each cell is `status / median_total_seconds / dataframe_loc+database_loc / query_rows`. "
            f"Timings are medians of {benchmark_runs} verified run(s). "
            f"`query_rows` is the aggregate result count from {KG_WORKLOAD_REPEATS} KG workload passes."
        ),
        "",
        "| DataFrame \\ Database | " + " | ".join(dbs) + " |",
        "| --- | " + " | ".join("---:" for _ in dbs) + " |",
    ]
    for dataframe in dfs:
        cells = []
        for database in dbs:
            result = lookup[(dataframe, database)]
            seconds = "" if result.total_seconds is None else f"{result.total_seconds:.4f}s"
            rows = "" if result.query_rows is None else str(result.query_rows)
            cells.append(
                f"{result.status}<br>{seconds}<br>LOC {result.dataframe_loc}+{result.database_loc}<br>rows {rows}"
            )
        lines.append(f"| {dataframe} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Cell Details",
            "",
            "| DataFrame | Database | Status | Median Total Seconds | Median DF Seconds | Median DB Seconds | Runs | DF LOC | DB LOC | Movies | Query Rows | Notes |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for result in sorted(results, key=median_total_sort_key):
        total = "" if result.total_seconds is None else f"{result.total_seconds:.4f}"
        df_seconds = "" if result.dataframe_seconds is None else f"{result.dataframe_seconds:.4f}"
        db_seconds = "" if result.database_seconds is None else f"{result.database_seconds:.4f}"
        movies = "" if result.movies is None else str(result.movies)
        rows = "" if result.query_rows is None else str(result.query_rows)
        notes = markdown_cell(result.notes)
        lines.append(
            f"| {markdown_cell(result.dataframe)} | {markdown_cell(result.database)} | {markdown_cell(result.status)} | "
            f"{total} | {df_seconds} | {db_seconds} | "
            f"{result.benchmark_runs} | {result.dataframe_loc} | {result.database_loc} | {movies} | {rows} | {notes} |"
        )
    return "\n".join(lines) + "\n"


def code_line_count(func: Callable) -> int:
    source = inspect.getsource(func)
    return sum(1 for line in source.splitlines() if line.strip() and not line.strip().startswith("#"))


def adapter_code_line_count(kind: str, name: str, fallback_func: Callable) -> int:
    grouped = DATAFRAME_LOC_GROUPS if kind == "dataframe" else DATABASE_LOC_GROUPS
    items = grouped.get(name)
    if not items:
        return code_line_count(fallback_func)
    total = 0
    for item in items:
        if callable(item):
            total += code_line_count(item)
        else:
            total += text_line_count(str(item))
    return total


def text_line_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


DATAFRAME_LOC_GROUPS = {
    "lynxes": [build_with_lynxes_project],
    "pandas": [build_with_pandas, dataframe_rows_to_source_rows],
    "polars": [build_with_polars, dataframe_rows_to_source_rows],
    "dask": [build_with_dask, dataframe_rows_to_source_rows],
}

DATABASE_LOC_GROUPS = {
    "caracaldb": [
        query_with_caracaldb,
        run_caracaldb_native_kg_workload,
        required_caracal_node,
        count_caracal_collaborations,
        count_caracal_similar_movies,
    ],
    "sqlite": [
        query_with_sqlite,
        load_sqlite_kg_tables,
        run_sqlite_kg_workload,
        SQL_GENRE_COUNT,
        SQL_ACTOR_MOVIE_COUNT,
        SQL_DIRECTOR_MOVIE_COUNT,
        SQL_COLLABORATION_COUNT,
        SQL_RECOMMENDATION_COUNT,
    ],
    "postgresql": [
        query_with_postgresql,
        run_postgresql_kg_workload,
        POSTGRES_GENRE_COUNT,
        POSTGRES_ACTOR_MOVIE_COUNT,
        POSTGRES_DIRECTOR_MOVIE_COUNT,
        POSTGRES_COLLABORATION_COUNT,
        POSTGRES_RECOMMENDATION_COUNT,
    ],
    "neo4j": [
        query_with_neo4j,
        load_neo4j_kg_tables,
        run_neo4j_kg_workload,
        "\n".join(NEO4J_INDEX_STATEMENTS),
        NEO4J_GENRE_COUNT,
        NEO4J_ACTOR_MOVIE_COUNT,
        NEO4J_DIRECTOR_MOVIE_COUNT,
        NEO4J_COLLABORATION_COUNT,
        NEO4J_RECOMMENDATION_COUNT,
    ],
}


class MissingExternalService(RuntimeError):
    pass


class WorkloadResultMismatch(RuntimeError):
    pass
