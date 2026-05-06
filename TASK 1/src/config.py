from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOVIES = PROJECT_ROOT.parent / "data" / "tmdb_5000_movies.csv"
DEFAULT_CREDITS = PROJECT_ROOT.parent / "data" / "tmdb_5000_credits.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "processed" / "tmdb_kg"


@dataclass(frozen=True)
class PipelineConfig:
    movies_csv: Path = DEFAULT_MOVIES
    credits_csv: Path = DEFAULT_CREDITS
    output_dir: Path = DEFAULT_OUTPUT_DIR
    processed_dir: Path = DEFAULT_PROCESSED_DIR
    db_path: Path = DEFAULT_DB_PATH
    top_cast: int = 5

