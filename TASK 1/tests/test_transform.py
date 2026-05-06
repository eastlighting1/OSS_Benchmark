from src.ingest import lynxes_frame_to_rows, rows_to_lynxes_frame
from src.transform import build_tables, parse_json_list


def test_parse_json_list_handles_invalid_values():
    assert parse_json_list("") == []
    assert parse_json_list("not json") == []
    assert parse_json_list('[{"id": 1, "name": "Action"}]') == [{"id": 1, "name": "Action"}]


def test_raw_rows_round_trip_through_lynxes_nodeframe():
    rows = [{"id": "1", "title": "Example"}]
    frame = rows_to_lynxes_frame(rows, label="RawMovie")

    assert "RawMovie" in str(frame)
    assert lynxes_frame_to_rows(frame) == rows


def test_build_tables_extracts_entities_and_relationships():
    movies = [
        {
            "id": "10",
            "title": "Example",
            "release_date": "2020-01-01",
            "budget": "100",
            "revenue": "300",
            "overview": "A movie",
            "genres": '[{"id": 1, "name": "Action"}]',
            "keywords": '[{"id": 7, "name": "space"}]',
        },
        {
            "id": "10",
            "title": "Duplicate",
            "release_date": "",
            "budget": "0",
            "revenue": "0",
            "overview": "",
            "genres": "[]",
            "keywords": "[]",
        },
    ]
    credits = [
        {
            "movie_id": "10",
            "title": "Example",
            "cast": '[{"id": 2, "name": "Actor", "character": "Hero", "order": 0}]',
            "crew": '[{"id": 3, "name": "Director", "job": "Director"}]',
        }
    ]

    tables = build_tables(movies, credits, top_cast=5)

    assert len(tables.movies) == 1
    assert tables.persons == [{"person_id": 2, "name": "Actor"}, {"person_id": 3, "name": "Director"}]
    assert tables.genres == [{"genre_id": 1, "name": "Action"}]
    assert tables.keywords == [{"keyword_id": 7, "name": "space"}]
    assert tables.acted_in[0]["character_name"] == "Hero"
    assert tables.directed == [{"person_id": 3, "movie_id": 10}]
