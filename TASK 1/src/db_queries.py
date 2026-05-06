from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import caracaldb
from caracaldb.api import open_edge_store

from .models import KnowledgeGraphTables, Table


NODE_CLASSES = ("Movie", "Person", "Genre", "Keyword")
EDGE_TYPES = ("ACTED_IN", "DIRECTED", "HAS_GENRE", "HAS_KEYWORD")


def load_tables_from_caracaldb(db_path: Path) -> KnowledgeGraphTables:
    """Read all required entity and relationship tables from caracaldb storage."""
    db = caracaldb.connect(db_path, format="bundle", mode="ro")
    try:
        return KnowledgeGraphTables(
            movies=strip_caracal_columns(read_node_store(db, "Movie")),
            persons=strip_caracal_columns(read_node_store(db, "Person")),
            genres=strip_caracal_columns(read_node_store(db, "Genre")),
            keywords=strip_caracal_columns(read_node_store(db, "Keyword")),
            acted_in=strip_caracal_columns(read_edge_store(db, "ACTED_IN")),
            directed=strip_caracal_columns(read_edge_store(db, "DIRECTED")),
            movie_genres=strip_caracal_columns(read_edge_store(db, "HAS_GENRE")),
            movie_keywords=strip_caracal_columns(read_edge_store(db, "HAS_KEYWORD")),
        )
    finally:
        db.close()


def read_node_store(db, class_name: str) -> Table:
    return db.open_node_store(class_name).to_table().to_pylist()


def read_edge_store(db, edge_type: str) -> Table:
    store = open_edge_store(
        db._bundle,
        property_iri=f"caracaldb:local:{edge_type}",
        local_name=edge_type,
        create=False,
    )
    return store.to_table().to_pylist()


def strip_caracal_columns(rows: Table) -> Table:
    internal = {"nid", "eid", "node_id", "type", "src", "dst", "_cdb_gid"}
    return [{key: value for key, value in row.items() if key not in internal} for row in rows]


def db_find_movie_by_title(db_path: Path, title: str) -> Table:
    tables = load_tables_from_caracaldb(db_path)
    needle = title.casefold()
    return [movie for movie in tables.movies if str(movie["title"]).casefold() == needle]


def db_movies_by_actor(db_path: Path, actor_name: str, limit: int = 20) -> Table:
    tables = load_tables_from_caracaldb(db_path)
    person_ids = person_ids_by_name(tables, actor_name)
    movie_by_id = {row["movie_id"]: row for row in tables.movies}
    rows: Table = []
    for rel in tables.acted_in:
        if rel["person_id"] not in person_ids:
            continue
        movie = movie_by_id.get(rel["movie_id"])
        if movie:
            rows.append(
                {
                    "person_name": actor_name,
                    "movie_id": movie["movie_id"],
                    "title": movie["title"],
                    "character_name": rel.get("character_name", ""),
                    "cast_order": rel.get("cast_order", ""),
                    "release_date": movie.get("release_date", ""),
                    "revenue": movie.get("revenue", 0),
                }
            )
    return sorted(rows, key=lambda row: (int(row["cast_order"] or 999999), str(row["title"])))[:limit]


def db_movies_by_director(db_path: Path, director_name: str, limit: int = 20) -> Table:
    tables = load_tables_from_caracaldb(db_path)
    person_ids = person_ids_by_name(tables, director_name)
    movie_by_id = {row["movie_id"]: row for row in tables.movies}
    rows: Table = []
    for rel in tables.directed:
        if rel["person_id"] not in person_ids:
            continue
        movie = movie_by_id.get(rel["movie_id"])
        if movie:
            rows.append(
                {
                    "director_name": director_name,
                    "movie_id": movie["movie_id"],
                    "title": movie["title"],
                    "release_date": movie.get("release_date", ""),
                    "revenue": movie.get("revenue", 0),
                }
            )
    return sorted(rows, key=lambda row: str(row["title"]))[:limit]


def db_movies_by_genre(db_path: Path, genre_names: list[str], limit: int = 20) -> Table:
    tables = load_tables_from_caracaldb(db_path)
    requested = {genre.casefold() for genre in genre_names}
    genre_by_id = {row["genre_id"]: str(row["name"]) for row in tables.genres}
    movie_by_id = {row["movie_id"]: row for row in tables.movies}
    genres_by_movie: defaultdict[int, set[str]] = defaultdict(set)
    for rel in tables.movie_genres:
        genres_by_movie[int(rel["movie_id"])].add(genre_by_id[rel["genre_id"]])

    rows: Table = []
    for movie_id, names in genres_by_movie.items():
        if requested.issubset({name.casefold() for name in names}):
            movie = movie_by_id[movie_id]
            rows.append(
                {
                    "movie_id": movie_id,
                    "title": movie["title"],
                    "genres": "|".join(sorted(names)),
                    "release_date": movie.get("release_date", ""),
                    "revenue": movie.get("revenue", 0),
                }
            )
    return sorted(rows, key=lambda row: int(row["revenue"]), reverse=True)[:limit]


def person_ids_by_name(tables: KnowledgeGraphTables, name: str) -> set[int]:
    needle = name.casefold()
    return {int(row["person_id"]) for row in tables.persons if str(row["name"]).casefold() == needle}
