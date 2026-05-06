from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any

from .models import KnowledgeGraphTables, Table


def parse_json_list(value: str | None) -> list[dict[str, Any]]:
    """Parse TMDB JSON-like list columns; invalid or blank values become empty lists."""
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def to_int(value: object, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def clean_text(value: object) -> str:
    return str(value or "").strip()


def merge_movies_credits(
    movies_rows: list[dict[str, str]], credits_rows: list[dict[str, str]]
) -> list[dict[str, object]]:
    credits_by_movie_id = {
        to_int(row.get("movie_id")): row for row in credits_rows if to_int(row.get("movie_id")) > 0
    }
    merged: OrderedDict[int, dict[str, object]] = OrderedDict()
    for row in movies_rows:
        movie_id = to_int(row.get("id"))
        title = clean_text(row.get("title"))
        if movie_id <= 0 or not title or movie_id in merged:
            continue
        credit = credits_by_movie_id.get(movie_id, {})
        merged[movie_id] = {
            "movie_id": movie_id,
            "title": title,
            "release_date": clean_text(row.get("release_date")),
            "budget": to_int(row.get("budget")),
            "revenue": to_int(row.get("revenue")),
            "overview": clean_text(row.get("overview")),
            "genres": parse_json_list(row.get("genres")),
            "keywords": parse_json_list(row.get("keywords")),
            "cast": parse_json_list(credit.get("cast")),
            "crew": parse_json_list(credit.get("crew")),
        }
    return list(merged.values())


def build_tables(
    movies_rows: list[dict[str, str]], credits_rows: list[dict[str, str]], top_cast: int = 5
) -> KnowledgeGraphTables:
    merged = merge_movies_credits(movies_rows, credits_rows)
    persons: OrderedDict[int, dict[str, object]] = OrderedDict()
    genres: OrderedDict[int, dict[str, object]] = OrderedDict()
    keywords: OrderedDict[int, dict[str, object]] = OrderedDict()
    tables = KnowledgeGraphTables()

    for movie in merged:
        movie_id = int(movie["movie_id"])
        tables.movies.append(
            {
                "movie_id": movie_id,
                "title": movie["title"],
                "release_date": movie["release_date"],
                "budget": movie["budget"],
                "revenue": movie["revenue"],
                "overview": movie["overview"],
            }
        )

        for genre in movie["genres"]:
            genre_id = to_int(genre.get("id"))
            name = clean_text(genre.get("name"))
            if genre_id <= 0 or not name:
                continue
            genres.setdefault(genre_id, {"genre_id": genre_id, "name": name})
            tables.movie_genres.append({"movie_id": movie_id, "genre_id": genre_id})

        for keyword in movie["keywords"]:
            keyword_id = to_int(keyword.get("id"))
            name = clean_text(keyword.get("name"))
            if keyword_id <= 0 or not name:
                continue
            keywords.setdefault(keyword_id, {"keyword_id": keyword_id, "name": name})
            tables.movie_keywords.append({"movie_id": movie_id, "keyword_id": keyword_id})

        cast_items = sorted(movie["cast"], key=lambda item: to_int(item.get("order"), 999999))[:top_cast]
        for cast_member in cast_items:
            person_id = to_int(cast_member.get("id"))
            name = clean_text(cast_member.get("name"))
            if person_id <= 0 or not name:
                continue
            persons.setdefault(person_id, {"person_id": person_id, "name": name})
            tables.acted_in.append(
                {
                    "person_id": person_id,
                    "movie_id": movie_id,
                    "character_name": clean_text(cast_member.get("character")),
                    "cast_order": to_int(cast_member.get("order")),
                }
            )

        for crew_member in movie["crew"]:
            if crew_member.get("job") != "Director":
                continue
            person_id = to_int(crew_member.get("id"))
            name = clean_text(crew_member.get("name"))
            if person_id <= 0 or not name:
                continue
            persons.setdefault(person_id, {"person_id": person_id, "name": name})
            tables.directed.append({"person_id": person_id, "movie_id": movie_id})

    tables.persons = list(persons.values())
    tables.genres = list(genres.values())
    tables.keywords = list(keywords.values())
    tables.acted_in = unique_rows(tables.acted_in, ["person_id", "movie_id"])
    tables.directed = unique_rows(tables.directed, ["person_id", "movie_id"])
    tables.movie_genres = unique_rows(tables.movie_genres, ["movie_id", "genre_id"])
    tables.movie_keywords = unique_rows(tables.movie_keywords, ["movie_id", "keyword_id"])
    return tables


def unique_rows(rows: Table, key_fields: list[str]) -> Table:
    seen: set[tuple[object, ...]] = set()
    result: Table = []
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result

