import pytest

from src.models import KnowledgeGraphTables
from src.schema import validate_tables


def test_validate_tables_accepts_valid_relationships():
    tables = KnowledgeGraphTables(
        movies=[{"movie_id": 1, "title": "A"}],
        persons=[{"person_id": 2, "name": "Actor"}],
        genres=[{"genre_id": 3, "name": "Action"}],
        keywords=[{"keyword_id": 4, "name": "space"}],
        acted_in=[{"person_id": 2, "movie_id": 1}],
        movie_genres=[{"movie_id": 1, "genre_id": 3}],
        movie_keywords=[{"movie_id": 1, "keyword_id": 4}],
    )

    validate_tables(tables)


def test_validate_tables_rejects_missing_entity_reference():
    tables = KnowledgeGraphTables(
        movies=[{"movie_id": 1, "title": "A"}],
        persons=[],
        acted_in=[{"person_id": 2, "movie_id": 1}],
    )

    with pytest.raises(ValueError):
        validate_tables(tables)

