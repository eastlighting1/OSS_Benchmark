from __future__ import annotations

from collections import defaultdict

from .models import KnowledgeGraphTables, Table
from .queries import find_movie_by_title


def recommend_movies(
    tables: KnowledgeGraphTables, movie: str | int, limit: int = 10
) -> Table:
    base = movie_features(tables, movie)
    movie_by_id = {int(row["movie_id"]): row for row in tables.movies}
    rows: Table = []
    for movie_id, candidate in all_movie_features(tables).items():
        if movie_id == base.movie_id:
            continue
        common_genres = base.genres & candidate.genres
        common_keywords = base.keywords & candidate.keywords
        common_actors = base.actors & candidate.actors
        same_director = bool(base.directors & candidate.directors)
        score = (
            2.0 * len(common_genres)
            + 1.5 * len(common_keywords)
            + 2.0 * len(common_actors)
            + 3.0 * int(same_director)
        )
        if score <= 0:
            continue
        rows.append(
            {
                "base_movie": movie_by_id[base.movie_id]["title"],
                "recommended_movie": movie_by_id[movie_id]["title"],
                "similarity_score": score,
                "common_genres": "|".join(sorted(common_genres)),
                "common_keywords": "|".join(sorted(common_keywords)),
                "common_actors": "|".join(sorted(common_actors)),
                "same_director": same_director,
            }
        )
    return sorted(rows, key=lambda row: (-float(row["similarity_score"]), row["recommended_movie"]))[:limit]


class MovieFeature:
    def __init__(
        self,
        movie_id: int,
        genres: set[str],
        keywords: set[str],
        actors: set[str],
        directors: set[str],
    ) -> None:
        self.movie_id = movie_id
        self.genres = genres
        self.keywords = keywords
        self.actors = actors
        self.directors = directors


def movie_features(tables: KnowledgeGraphTables, movie: str | int) -> MovieFeature:
    if isinstance(movie, int) or str(movie).isdigit():
        movie_id = int(movie)
        if movie_id not in {int(row["movie_id"]) for row in tables.movies}:
            raise LookupError(f"Movie not found: {movie}")
    else:
        movie_id = int(find_movie_by_title(tables, str(movie))["movie_id"])
    return all_movie_features(tables)[movie_id]


def all_movie_features(tables: KnowledgeGraphTables) -> dict[int, MovieFeature]:
    genre_by_id = {int(row["genre_id"]): str(row["name"]) for row in tables.genres}
    keyword_by_id = {int(row["keyword_id"]): str(row["name"]) for row in tables.keywords}
    person_by_id = {int(row["person_id"]): str(row["name"]) for row in tables.persons}
    genres: defaultdict[int, set[str]] = defaultdict(set)
    keywords: defaultdict[int, set[str]] = defaultdict(set)
    actors: defaultdict[int, set[str]] = defaultdict(set)
    directors: defaultdict[int, set[str]] = defaultdict(set)

    for row in tables.movie_genres:
        genres[int(row["movie_id"])].add(genre_by_id[int(row["genre_id"])])
    for row in tables.movie_keywords:
        keywords[int(row["movie_id"])].add(keyword_by_id[int(row["keyword_id"])])
    for row in tables.acted_in:
        actors[int(row["movie_id"])].add(person_by_id[int(row["person_id"])])
    for row in tables.directed:
        directors[int(row["movie_id"])].add(person_by_id[int(row["person_id"])])

    return {
        int(row["movie_id"]): MovieFeature(
            int(row["movie_id"]),
            genres[int(row["movie_id"])],
            keywords[int(row["movie_id"])],
            actors[int(row["movie_id"])],
            directors[int(row["movie_id"])],
        )
        for row in tables.movies
    }

