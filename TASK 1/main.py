from __future__ import annotations

import argparse
from pathlib import Path

from src.compare_systems import run_comparison
from src.config import PipelineConfig
from src.db_queries import (
    db_find_movie_by_title,
    db_movies_by_actor,
    db_movies_by_director,
    db_movies_by_genre,
    load_tables_from_caracaldb,
)
from src.pipeline import build_pipeline, print_table
from src.queries import director_actor_collaborations
from src.recommender import recommend_movies
from src.storage import write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TMDB knowledge graph pipeline using caracaldb and lynxes.")
    parser.add_argument("command", nargs="?", default="run", choices=[
        "run",
        "ingest",
        "build-kg",
        "query-movie",
        "query-actor",
        "query-director",
        "query-collaborations",
        "query-genre",
        "recommend",
        "compare",
    ])
    parser.add_argument("--movies", type=Path, default=None)
    parser.add_argument("--credits", type=Path, default=None)
    parser.add_argument("--movie", default="Interstellar")
    parser.add_argument("--actor", default="Tom Hanks")
    parser.add_argument("--director", default=None)
    parser.add_argument("--genres", default="Science Fiction")
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--top-cast", type=int, default=5)
    parser.add_argument("--benchmark-runs", type=int, default=3)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> PipelineConfig:
    base = PipelineConfig()
    return PipelineConfig(
        movies_csv=args.movies or base.movies_csv,
        credits_csv=args.credits or base.credits_csv,
        output_dir=base.output_dir,
        processed_dir=base.processed_dir,
        db_path=base.db_path,
        top_cast=args.top_cast,
    )


def main() -> None:
    args = parse_args()
    config = make_config(args)

    if args.command in {"run", "ingest", "build-kg"}:
        tables = build_pipeline(config)
        print("Pipeline complete.")
        for name, count in tables.counts().items():
            print(f"{name}: {count}")
        print(f"Processed tables: {config.processed_dir}")
        print(f"Reports: {config.output_dir}")
        return

    if args.command == "query-collaborations":
        tables = load_tables_from_caracaldb(config.db_path)
        rows = director_actor_collaborations(
            tables,
            min_count=args.min_count,
            limit=args.limit,
            director_name=args.director,
        )
        write_csv(config.output_dir / "collaboration_report.csv", rows)
        print_table(rows, limit=args.limit)
        return

    if args.command == "query-movie":
        rows = db_find_movie_by_title(config.db_path, args.movie)
        print_table(rows, limit=args.limit)
        return

    if args.command == "query-actor":
        rows = db_movies_by_actor(config.db_path, args.actor, limit=args.limit)
        print_table(rows, limit=args.limit)
        return

    if args.command == "query-director":
        if not args.director:
            raise SystemExit("--director is required for query-director")
        rows = db_movies_by_director(config.db_path, args.director, limit=args.limit)
        print_table(rows, limit=args.limit)
        return

    if args.command == "query-genre":
        genres = [item.strip() for item in args.genres.split(",") if item.strip()]
        rows = db_movies_by_genre(config.db_path, genres, limit=args.limit)
        write_csv(config.output_dir / "genre_query_report.csv", rows)
        print_table(rows, limit=args.limit)
        return

    if args.command == "recommend":
        tables = load_tables_from_caracaldb(config.db_path)
        rows = recommend_movies(tables, args.movie, limit=args.limit)
        write_csv(config.output_dir / "recommendation_report.csv", rows)
        print_table(rows, limit=args.limit)
        return

    if args.command == "compare":
        rows = [result.__dict__ for result in run_comparison(config, benchmark_runs=args.benchmark_runs)]
        print_table(rows, limit=len(rows))


if __name__ == "__main__":
    main()
