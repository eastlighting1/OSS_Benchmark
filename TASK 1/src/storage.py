from __future__ import annotations

import csv
import shutil
from pathlib import Path

import caracaldb
import pyarrow as pa

from .models import KnowledgeGraphTables, Table


def write_processed_tables(tables: KnowledgeGraphTables, processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.table_map().items():
        write_csv(processed_dir / f"{name}.csv", rows)


def write_csv(path: Path, rows: Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_into_caracaldb(tables: KnowledgeGraphTables, db_path: Path) -> dict[str, int]:
    """Store normalized entities and relationships in caracaldb.

    The load is rebuilt from scratch for reproducibility and to avoid duplicate rows
    when the same pipeline is run repeatedly.
    """
    remove_existing_database(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = caracaldb.connect(db_path, format="bundle")
    try:
        db.insert_node_table_arrow(rows_to_arrow_table(entity_rows_for_db(tables)), key_col="node_id", type_col="type")
        db.insert_edge_table_arrow(
            rows_to_arrow_table(edge_rows_for_db(tables)),
            src_col="src",
            dst_col="dst",
            type_col="type",
        )
    finally:
        db.close()
    return tables.counts()


def remove_existing_database(db_path: Path) -> None:
    candidates = [db_path, db_path.with_suffix(".crcl")]
    for candidate in candidates:
        if candidate.is_dir():
            shutil.rmtree(candidate)
        elif candidate.exists():
            candidate.unlink()


def entity_rows_for_db(tables: KnowledgeGraphTables) -> Table:
    rows: Table = []
    rows.extend(
        {
            "node_id": f"movie:{row['movie_id']}",
            "type": "Movie",
            **row,
        }
        for row in tables.movies
    )
    rows.extend(
        {
            "node_id": f"person:{row['person_id']}",
            "type": "Person",
            **row,
        }
        for row in tables.persons
    )
    rows.extend(
        {
            "node_id": f"genre:{row['genre_id']}",
            "type": "Genre",
            **row,
        }
        for row in tables.genres
    )
    rows.extend(
        {
            "node_id": f"keyword:{row['keyword_id']}",
            "type": "Keyword",
            **row,
        }
        for row in tables.keywords
    )
    return rows


def edge_rows_for_db(tables: KnowledgeGraphTables) -> Table:
    rows: Table = []
    rows.extend(
        {
            "src": f"person:{row['person_id']}",
            "dst": f"movie:{row['movie_id']}",
            "type": "ACTED_IN",
            **row,
        }
        for row in tables.acted_in
    )
    rows.extend(
        {
            "src": f"person:{row['person_id']}",
            "dst": f"movie:{row['movie_id']}",
            "type": "DIRECTED",
            **row,
        }
        for row in tables.directed
    )
    rows.extend(
        {
            "src": f"movie:{row['movie_id']}",
            "dst": f"genre:{row['genre_id']}",
            "type": "HAS_GENRE",
            **row,
        }
        for row in tables.movie_genres
    )
    rows.extend(
        {
            "src": f"movie:{row['movie_id']}",
            "dst": f"keyword:{row['keyword_id']}",
            "type": "HAS_KEYWORD",
            **row,
        }
        for row in tables.movie_keywords
    )
    return rows


def rows_to_arrow_table(rows: Table) -> pa.Table:
    if not rows:
        return pa.table({})
    columns = sorted({key for row in rows for key in row})
    normalized = [{column: row.get(column) for column in columns} for row in rows]
    return pa.Table.from_pylist(normalized)
