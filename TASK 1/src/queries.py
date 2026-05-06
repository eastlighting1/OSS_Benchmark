from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import caracaldb

from .models import KnowledgeGraphTables, Table


def find_movie_by_title(tables: KnowledgeGraphTables, title: str) -> dict[str, object]:
    needle = title.casefold()
    for movie in tables.movies:
        if str(movie["title"]).casefold() == needle:
            return movie
    raise LookupError(f"Movie not found: {title}")


def find_movies_by_genres(tables: KnowledgeGraphTables, genres: list[str], limit: int = 20) -> Table:
    requested = {genre.casefold() for genre in genres}
    genre_by_id = {row["genre_id"]: row["name"] for row in tables.genres}
    movie_by_id = {row["movie_id"]: row for row in tables.movies}
    genres_by_movie: defaultdict[int, set[str]] = defaultdict(set)
    for row in tables.movie_genres:
        genres_by_movie[int(row["movie_id"])].add(str(genre_by_id[row["genre_id"]]))

    matches: Table = []
    for movie_id, names in genres_by_movie.items():
        if requested.issubset({name.casefold() for name in names}):
            movie = movie_by_id[movie_id]
            matches.append(
                {
                    "movie_id": movie_id,
                    "title": movie["title"],
                    "genres": "|".join(sorted(names)),
                    "release_date": movie["release_date"],
                    "revenue": movie["revenue"],
                }
            )
    return sorted(matches, key=lambda row: int(row["revenue"]), reverse=True)[:limit]


def director_actor_collaborations(
    tables: KnowledgeGraphTables, min_count: int = 2, limit: int = 20, director_name: str | None = None
) -> Table:
    person_by_id = {row["person_id"]: row["name"] for row in tables.persons}
    movie_by_id = {row["movie_id"]: row["title"] for row in tables.movies}
    directors_by_movie: defaultdict[int, set[int]] = defaultdict(set)
    actors_by_movie: defaultdict[int, set[int]] = defaultdict(set)

    for row in tables.directed:
        directors_by_movie[int(row["movie_id"])].add(int(row["person_id"]))
    for row in tables.acted_in:
        actors_by_movie[int(row["movie_id"])].add(int(row["person_id"]))

    pairs: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for movie_id, directors in directors_by_movie.items():
        for director_id in directors:
            for actor_id in actors_by_movie.get(movie_id, set()):
                if actor_id != director_id:
                    pairs[(director_id, actor_id)].append(movie_id)

    rows: Table = []
    director_filter = director_name.casefold() if director_name else None
    for (director_id, actor_id), movie_ids in pairs.items():
        if len(movie_ids) < min_count:
            continue
        resolved_director = str(person_by_id.get(director_id, str(director_id)))
        if director_filter and resolved_director.casefold() != director_filter:
            continue
        rows.append(
            {
                "director_name": resolved_director,
                "actor_name": person_by_id.get(actor_id, str(actor_id)),
                "collaboration_count": len(movie_ids),
                "movie_titles": "|".join(movie_by_id[movie_id] for movie_id in sorted(movie_ids)),
            }
        )
    return sorted(rows, key=lambda row: (-int(row["collaboration_count"]), row["director_name"], row["actor_name"]))[
        :limit
    ]


def genre_report(tables: KnowledgeGraphTables) -> Table:
    genre_by_id = {row["genre_id"]: row["name"] for row in tables.genres}
    counter = Counter(row["genre_id"] for row in tables.movie_genres)
    revenue_by_genre: defaultdict[int, int] = defaultdict(int)
    movie_by_id = {row["movie_id"]: row for row in tables.movies}
    for row in tables.movie_genres:
        revenue_by_genre[int(row["genre_id"])] += int(movie_by_id[row["movie_id"]]["revenue"])
    return [
        {
            "genre_name": genre_by_id[genre_id],
            "movie_count": count,
            "total_revenue": revenue_by_genre[genre_id],
        }
        for genre_id, count in counter.most_common()
    ]


def central_people(tables: KnowledgeGraphTables, limit: int = 20) -> Table:
    person_by_id = {int(row["person_id"]): str(row["name"]) for row in tables.persons}
    acted_by_person = Counter(int(row["person_id"]) for row in tables.acted_in)
    directed_by_person = Counter(int(row["person_id"]) for row in tables.directed)
    actors_by_movie: defaultdict[int, set[int]] = defaultdict(set)
    for row in tables.acted_in:
        actors_by_movie[int(row["movie_id"])].add(int(row["person_id"]))

    collaborators: defaultdict[int, set[int]] = defaultdict(set)
    for actor_ids in actors_by_movie.values():
        for actor_id in actor_ids:
            collaborators[actor_id].update(actor_ids - {actor_id})

    rows: Table = []
    for person_id, name in person_by_id.items():
        acted = acted_by_person[person_id]
        directed = directed_by_person[person_id]
        unique_collaborators = len(collaborators[person_id])
        score = acted + (2 * directed) + (0.25 * unique_collaborators)
        if score <= 0:
            continue
        rows.append(
            {
                "person_name": name,
                "acted_movie_count": acted,
                "directed_movie_count": directed,
                "unique_collaborator_count": unique_collaborators,
                "centrality_score": round(score, 2),
            }
        )
    return sorted(rows, key=lambda row: (-float(row["centrality_score"]), row["person_name"]))[:limit]


def database_healthcheck(db_path: Path) -> int:
    db = caracaldb.connect(db_path, format="bundle", mode="ro")
    try:
        result = db.sql("MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p,m LIMIT 1")
        return len(result.rows())
    finally:
        db.close()


def load_processed_tables(processed_dir: Path) -> KnowledgeGraphTables:
    return KnowledgeGraphTables(
        movies=read_csv(processed_dir / "movies.csv"),
        persons=read_csv(processed_dir / "persons.csv"),
        genres=read_csv(processed_dir / "genres.csv"),
        keywords=read_csv(processed_dir / "keywords.csv"),
        acted_in=read_csv(processed_dir / "acted_in.csv"),
        directed=read_csv(processed_dir / "directed.csv"),
        movie_genres=read_csv(processed_dir / "movie_genres.csv"),
        movie_keywords=read_csv(processed_dir / "movie_keywords.csv"),
    )


def read_csv(path: Path) -> Table:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [coerce_row(row) for row in csv.DictReader(handle)]


def coerce_row(row: dict[str, str]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in row.items():
        if key.endswith("_id") or key in {"budget", "revenue", "cast_order"}:
            result[key] = int(value or 0)
        else:
            result[key] = value
    return result
