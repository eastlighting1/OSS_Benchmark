import duckdb
import torch
import numpy as np
import pyarrow as pa
from typing import List
from .base import BaseBackend
from ..config import BenchmarkConfig

class DuckDBBackend(BaseBackend):
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.db_path = config.data_dir / "processed" / "benchmark_duckdb.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self.node_features = None

    def ingest(self, node_feat_df, node_label_df, node_year_df, edge_df):
        print(f"Ingesting data into DuckDB at {self.db_path}...")
        import pandas as pd
        # Clear old data
        self.conn.execute("DROP TABLE IF EXISTS edges")
        if hasattr(edge_df, "to_arrow"):
            # lynxes / arrow-backed
            raw_arrow = edge_df.to_arrow()
            edge_table = pa.Table.from_batches([raw_arrow]) if isinstance(raw_arrow, pa.RecordBatch) else raw_arrow
            cols = edge_table.column_names
            data_cols = [c for c in cols if not str(c).startswith("_")]
            edge_table = edge_table.select(data_cols)
            edge_table = edge_table.rename_columns(["src", "dst"])
        elif isinstance(edge_df, pd.DataFrame):
            edge_table = edge_df[[c for c in edge_df.columns if not str(c).startswith("_")]]
            edge_table.columns = ["src", "dst"]
        else:
            # Fallback for polars/dask via compute
            pdf = edge_df.compute() if hasattr(edge_df, "compute") else edge_df.to_pandas()
            edge_table = pdf[[c for c in pdf.columns if not str(c).startswith("_")]]
            edge_table.columns = ["src", "dst"]
        
        self.conn.execute("CREATE TABLE edges AS SELECT * FROM edge_table")
        self.node_features = node_feat_df
        
    def get_sampler(self, fanouts: List[int], batch_size: int, scenario: str = "default", filter_data: Any = None, num_workers: int = 0):
        from ..samplers.duckdb_sampler import DuckDBSampler
        return DuckDBSampler(self.conn, self.node_features, fanouts, batch_size, scenario=scenario, filter_data=filter_data)

    def fetch_features(self, node_ids: torch.Tensor) -> torch.Tensor:
        indices = node_ids.tolist()
        # Handle different DF types for gather_rows
        if hasattr(self.node_features, "gather_rows"):
            batch_feat_df = self.node_features.gather_rows(indices)
        elif hasattr(self.node_features, "iloc"):
            # pandas
            batch_feat_df = self.node_features.iloc[indices]
        else:
            # polars/dask
            batch_feat_df = self.node_features[indices]
        
        from ..ingest import convert_to_tensor
        return convert_to_tensor(batch_feat_df)
