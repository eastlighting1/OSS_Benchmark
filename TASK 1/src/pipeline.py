from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig
from .compare_systems import run_comparison
from .ingest import load_source_data
from .kg import build_lynxes_graph, write_graph_artifacts
from .models import KnowledgeGraphTables, Table
from .queries import central_people, database_healthcheck, director_actor_collaborations, genre_report
from .recommender import recommend_movies
from .schema import validate_tables
from .storage import load_into_caracaldb, write_csv, write_processed_tables
from .transform import build_tables


def build_pipeline(config: PipelineConfig) -> KnowledgeGraphTables:
    movies_rows, credits_rows = load_source_data(config.movies_csv, config.credits_csv)
    tables = build_tables(movies_rows, credits_rows, top_cast=config.top_cast)
    validate_tables(tables)
    write_processed_tables(tables, config.processed_dir)
    load_into_caracaldb(tables, config.db_path)

    graph = build_lynxes_graph(tables)
    lynxes_summary = "\n\n".join(
        [
            graph.info(),
            graph.describe(),
            str(graph.pagerank(max_iter=20).head(20)),
            str(graph.degree_centrality().head(20)),
        ]
    )
    write_graph_artifacts(graph, config.output_dir)
    (config.output_dir / "lynxes_graph_summary.txt").write_text(lynxes_summary, encoding="utf-8")

    write_reports(tables, config.output_dir)
    run_comparison(config)
    return tables


def write_reports(tables: KnowledgeGraphTables, output_dir: Path, recommendation_movie: str = "Interstellar") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "collaboration_report.csv", director_actor_collaborations(tables))
    write_csv(output_dir / "genre_report.csv", genre_report(tables))
    write_csv(output_dir / "central_people_report.csv", central_people(tables))
    try:
        recommendations = recommend_movies(tables, recommendation_movie)
    except LookupError:
        recommendations = []
    write_csv(output_dir / "recommendation_report.csv", recommendations)


def print_table(rows: Table, limit: int = 10) -> None:
    if not rows:
        print("(no rows)")
        return
    columns = list(rows[0].keys())
    widths = {
        column: min(48, max(len(column), *(len(str(row[column])) for row in rows[:limit])))
        for column in columns
    }
    print(" | ".join(column.ljust(widths[column]) for column in columns))
    print("-+-".join("-" * widths[column] for column in columns))
    for row in rows[:limit]:
        values = [truncate(str(row[column]), widths[column]).ljust(widths[column]) for column in columns]
        print(" | ".join(values))


def truncate(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    return value[: max(0, width - 3)] + "..."


def assert_db_ready(config: PipelineConfig) -> None:
    count = database_healthcheck(config.db_path)
    if count < 0:
        raise RuntimeError("caracaldb healthcheck failed")
