from __future__ import annotations

from pathlib import Path

import lynxes

from .models import KnowledgeGraphTables
from .storage import edge_rows_for_db


def build_lynxes_graph(tables: KnowledgeGraphTables):
    """Create a lynxes GraphFrame from normalized KG tables."""
    node_rows = []
    for row in tables.movies:
        node_rows.append(
            {
                "_id": f"movie:{row['movie_id']}",
                "_label": ["Movie"],
                "name": str(row["title"]),
                "entity_id": int(row["movie_id"]),
            }
        )
    for row in tables.persons:
        node_rows.append(
            {
                "_id": f"person:{row['person_id']}",
                "_label": ["Person"],
                "name": str(row["name"]),
                "entity_id": int(row["person_id"]),
            }
        )
    for row in tables.genres:
        node_rows.append(
            {
                "_id": f"genre:{row['genre_id']}",
                "_label": ["Genre"],
                "name": str(row["name"]),
                "entity_id": int(row["genre_id"]),
            }
        )
    for row in tables.keywords:
        node_rows.append(
            {
                "_id": f"keyword:{row['keyword_id']}",
                "_label": ["Keyword"],
                "name": str(row["name"]),
                "entity_id": int(row["keyword_id"]),
            }
        )

    edge_rows = [
        {
            "_src": str(row["src"]),
            "_dst": str(row["dst"]),
            "_type": str(row["type"]),
            "_direction": 1,
        }
        for row in edge_rows_for_db(tables)
    ]
    return lynxes.graph(nodes=columns_from_rows(node_rows), edges=columns_from_rows(edge_rows))


def columns_from_rows(rows: list[dict[str, object]]) -> dict[str, list[object]]:
    if not rows:
        return {}
    columns = {key: [] for key in rows[0]}
    for row in rows:
        for key in columns:
            columns[key].append(row[key])
    return columns


def write_graph_artifacts(graph, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    graph.write_gf(str(output_dir / "tmdb_kg.gf"))

