from src.models import KnowledgeGraphTables
from src.db_queries import db_movies_by_actor, db_movies_by_director, load_tables_from_caracaldb
from src.queries import director_actor_collaborations, find_movies_by_genres
from src.recommender import recommend_movies
from src.storage import load_into_caracaldb


def sample_tables():
    return KnowledgeGraphTables(
        movies=[
            {"movie_id": 1, "title": "Base", "release_date": "2020", "revenue": 100},
            {"movie_id": 2, "title": "Close", "release_date": "2021", "revenue": 200},
            {"movie_id": 3, "title": "Far", "release_date": "2022", "revenue": 50},
        ],
        persons=[
            {"person_id": 10, "name": "Director"},
            {"person_id": 11, "name": "Actor"},
            {"person_id": 12, "name": "Other"},
        ],
        genres=[
            {"genre_id": 20, "name": "Science Fiction"},
            {"genre_id": 21, "name": "Action"},
        ],
        keywords=[{"keyword_id": 30, "name": "space"}],
        acted_in=[
            {"person_id": 11, "movie_id": 1},
            {"person_id": 11, "movie_id": 2},
            {"person_id": 12, "movie_id": 3},
        ],
        directed=[
            {"person_id": 10, "movie_id": 1},
            {"person_id": 10, "movie_id": 2},
        ],
        movie_genres=[
            {"movie_id": 1, "genre_id": 20},
            {"movie_id": 1, "genre_id": 21},
            {"movie_id": 2, "genre_id": 20},
            {"movie_id": 3, "genre_id": 21},
        ],
        movie_keywords=[
            {"movie_id": 1, "keyword_id": 30},
            {"movie_id": 2, "keyword_id": 30},
        ],
    )


def test_collaboration_analysis_sorts_by_count():
    rows = director_actor_collaborations(sample_tables(), min_count=2)
    assert rows[0]["director_name"] == "Director"
    assert rows[0]["actor_name"] == "Actor"
    assert rows[0]["collaboration_count"] == 2


def test_genre_search_requires_all_requested_genres():
    rows = find_movies_by_genres(sample_tables(), ["Science Fiction", "Action"])
    assert [row["title"] for row in rows] == ["Base"]


def test_recommendation_score_explains_overlap():
    rows = recommend_movies(sample_tables(), "Base")
    assert rows[0]["recommended_movie"] == "Close"
    assert rows[0]["similarity_score"] == 8.5
    assert rows[0]["same_director"] is True


def test_caracaldb_query_layer_reads_stored_entities(tmp_path):
    db_path = tmp_path / "mini_kg"
    tables = sample_tables()
    load_into_caracaldb(tables, db_path)

    loaded = load_tables_from_caracaldb(db_path)
    assert len(loaded.movies) == 3
    assert db_movies_by_actor(db_path, "Actor")[0]["title"] == "Base"
    assert db_movies_by_director(db_path, "Director")[0]["title"] == "Base"
