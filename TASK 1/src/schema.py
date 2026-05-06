from __future__ import annotations

from .models import KnowledgeGraphTables


def validate_tables(tables: KnowledgeGraphTables) -> None:
    """Validate required KG referential integrity constraints."""
    movie_ids = {row["movie_id"] for row in tables.movies}
    person_ids = {row["person_id"] for row in tables.persons}
    genre_ids = {row["genre_id"] for row in tables.genres}
    keyword_ids = {row["keyword_id"] for row in tables.keywords}

    for row in tables.acted_in:
        if row["person_id"] not in person_ids or row["movie_id"] not in movie_ids:
            raise ValueError(f"Invalid acted_in relationship: {row}")
    for row in tables.directed:
        if row["person_id"] not in person_ids or row["movie_id"] not in movie_ids:
            raise ValueError(f"Invalid directed relationship: {row}")
    for row in tables.movie_genres:
        if row["movie_id"] not in movie_ids or row["genre_id"] not in genre_ids:
            raise ValueError(f"Invalid movie_genres relationship: {row}")
    for row in tables.movie_keywords:
        if row["movie_id"] not in movie_ids or row["keyword_id"] not in keyword_ids:
            raise ValueError(f"Invalid movie_keywords relationship: {row}")

