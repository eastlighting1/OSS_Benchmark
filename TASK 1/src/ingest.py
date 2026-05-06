from __future__ import annotations

import csv
from pathlib import Path

import lynxes
import pyarrow.csv as pyarrow_csv


MOVIE_COLUMNS = (
    "id",
    "title",
    "release_date",
    "budget",
    "revenue",
    "overview",
    "genres",
    "keywords",
)
CREDIT_COLUMNS = ("movie_id", "title", "cast", "crew")
REQUIRED_MOVIE_COLUMNS = set(MOVIE_COLUMNS)
REQUIRED_CREDIT_COLUMNS = set(CREDIT_COLUMNS)


def load_csv_rows(
    path: Path,
    required_columns: set[str],
    label: str = "RawRow",
    id_col: str | None = None,
) -> list[dict[str, object]]:
    """Load CSV through lynxes, then return only the projected row dictionaries for ETL."""
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    projection = sorted(required_columns | ({id_col} if id_col else set()))
    try:
        frame = lynxes.read_csv(
            path,
            label=label,
            id_col=id_col,
            id_prefix=None if id_col else label.lower(),
            engine="pyarrow",
            convert_options=pyarrow_csv.ConvertOptions(include_columns=projection),
        )
    except Exception as exc:
        raise ValueError(f"CSV schema mismatch or unreadable CSV in {path}: {exc}") from exc
    missing = required_columns - set(frame.column_names())
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"CSV schema mismatch in {path}: missing {missing_list}")
    return lynxes_frame_to_rows(frame)


def load_csv_rows_with_stdlib(path: Path, required_columns: set[str], label: str = "RawRow") -> list[dict[str, object]]:
    """Fallback CSV loader retained for tests and constrained environments."""
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = required_columns - set(reader.fieldnames)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"CSV schema mismatch in {path}: missing {missing_list}")
        rows = list(reader)
    frame = rows_to_lynxes_frame(rows, label=label)
    if frame.len() != len(rows):
        raise ValueError(f"lynxes load row count mismatch for {path}")
    return rows


def load_source_data(movies_csv: Path, credits_csv: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    return (
        load_csv_rows(movies_csv, REQUIRED_MOVIE_COLUMNS, label="RawMovie", id_col="id"),
        load_csv_rows(credits_csv, REQUIRED_CREDIT_COLUMNS, label="RawCredit", id_col="movie_id"),
    )


def rows_to_lynxes_frame(rows: list[dict[str, str]], label: str):
    """Represent raw CSV rows as a lynxes NodeFrame for the processing layer."""
    columns: dict[str, list[object]] = {
        "_id": [f"{label.lower()}:{index}" for index in range(len(rows))],
        "_label": [[label] for _ in rows],
    }
    fieldnames = sorted({field for row in rows for field in row})
    for field in fieldnames:
        columns[field] = [row.get(field, "") for row in rows]
    return lynxes.NodeFrame.from_dict(columns)


def lynxes_frame_to_rows(frame) -> list[dict[str, object]]:
    """Convert a lynxes NodeFrame back to dictionaries after validation/loading."""
    columns = [column for column in frame.column_names() if not column.startswith("_")]
    values = {column: frame.column_values(column) for column in columns}
    row_count = frame.len()
    return [
        {column: (values[column][index] if values[column][index] is not None else "") for column in columns}
        for index in range(row_count)
    ]
