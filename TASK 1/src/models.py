from __future__ import annotations

from dataclasses import dataclass, field


Table = list[dict[str, object]]


@dataclass
class KnowledgeGraphTables:
    movies: Table = field(default_factory=list)
    persons: Table = field(default_factory=list)
    genres: Table = field(default_factory=list)
    keywords: Table = field(default_factory=list)
    acted_in: Table = field(default_factory=list)
    directed: Table = field(default_factory=list)
    movie_genres: Table = field(default_factory=list)
    movie_keywords: Table = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {
            "movies": len(self.movies),
            "persons": len(self.persons),
            "genres": len(self.genres),
            "keywords": len(self.keywords),
            "acted_in": len(self.acted_in),
            "directed": len(self.directed),
            "movie_genres": len(self.movie_genres),
            "movie_keywords": len(self.movie_keywords),
        }

    def table_map(self) -> dict[str, Table]:
        return {
            "movies": self.movies,
            "persons": self.persons,
            "genres": self.genres,
            "keywords": self.keywords,
            "acted_in": self.acted_in,
            "directed": self.directed,
            "movie_genres": self.movie_genres,
            "movie_keywords": self.movie_keywords,
        }

