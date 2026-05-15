import caracaldb
import lynxes as lx
import torch
import pyarrow as pa
import numpy as np
from typing import List, Optional, Any
from .base import BaseBackend
from ..config import BenchmarkConfig

class CaracalLynxesBackend(BaseBackend):
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.db_path = config.data_dir / "processed" / f"{config.caracal_db_name}.crcl"
        self._db = None
        self.node_features = None # Store as lynxes frame
        self.node_labels = None
        
    def ingest(self, node_feat_df, node_label_df, node_year_df, edge_df):
        from ..ingest import get_df_len, get_df_cols
        print(f"Ingesting data into caracaldb at {self.db_path}")
        
        # Preserve frames for sampling/fetching
        self.node_features = node_feat_df
        self.node_labels = node_label_df

        # Handle Windows file locks: avoid full rmtree if possible
        # Or use a new path for each ingestion
        import time
        ingest_db_path = self.db_path
        
        # If we can't delete it, it means it's locked.
        # For the benchmark, we can just use a unique name if locked.
        if ingest_db_path.exists():
            try:
                import shutil
                if ingest_db_path.is_dir():
                    shutil.rmtree(ingest_db_path)
                else:
                    ingest_db_path.unlink()
            except Exception as e:
                print(f"Warning: Could not delete locked DB at {ingest_db_path}: {e}")
                # Use a timestamped path to bypass lock
                ingest_db_path = ingest_db_path.with_name(f"{ingest_db_path.stem}_{int(time.time()*1000)}.crcl")
                self.db_path = ingest_db_path
                print(f"Using alternate path: {self.db_path}")
        
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        db = caracaldb.connect(self.db_path, format="bundle")
        
        try:
            # 1. Ingest Nodes with Int64 IDs and properties
            num_nodes = get_df_len(node_feat_df)
            node_ids = np.arange(num_nodes, dtype=np.int64)
            
            # Prepare properties table from node_year_df
            # ENSURE DASK IS COMPUTED
            pdf_year = node_year_df.compute() if hasattr(node_year_df, "compute") else node_year_df
            
            if hasattr(pdf_year, "to_arrow"):
                year_table = pdf_year.to_arrow()
            else:
                year_table = pa.Table.from_pandas(pdf_year)
            
            year_cols = [c for c in year_table.column_names if not str(c).startswith("_")]
            year_col = year_cols[0]
            
            node_table = pa.Table.from_pydict({
                "node_id": node_ids, 
                "type": ["Node"] * num_nodes,
                "year": year_table.column(year_col).cast(pa.int64())
            })
            db.insert_node_table_arrow(node_table, key_col="node_id", type_col="type")

            # 1.1 Create Property Index for Scenario B
            print("Creating property index on 'year'...")
            db.create_property_index(name="idx_node_year", node_type="Node", property="year")

            # 2. Prepare edge table (Int64)
            import pandas as pd
            pdf_edge = edge_df.compute() if hasattr(edge_df, "compute") else (edge_df.to_pandas() if hasattr(edge_df, "to_pandas") else edge_df)
            
            if hasattr(pdf_edge, "to_arrow"):
                raw_arrow = pdf_edge.to_arrow()
                edge_arrow = pa.Table.from_batches([raw_arrow]) if isinstance(raw_arrow, pa.RecordBatch) else raw_arrow
            else:
                edge_arrow = pa.Table.from_pandas(pdf_edge)
                
            # FILTER COLUMNS FIRST
            all_cols = edge_arrow.column_names
            data_cols = [c for c in all_cols if not str(c).startswith("_")]
            edge_arrow = edge_arrow.select(data_cols)
            
            edge_arrow = edge_arrow.rename_columns(["src", "dst"])
            edge_arrow = edge_arrow.cast(pa.schema([("src", pa.int64()), ("dst", pa.int64())]))
            edge_arrow = edge_arrow.append_column("type", pa.array(["CITES"] * edge_arrow.num_rows))
            
            db.insert_edge_table_arrow(edge_arrow, src_col="src", dst_col="dst", type_col="type")
            
        finally:
            db.close()

    def _get_db(self):
        if self._db is None:
            self._db = caracaldb.connect(self.db_path, format="bundle", mode="ro")
        return self._db

    def get_sampler(
        self,
        fanouts: List[int],
        batch_size: int,
        scenario: str = "default",
        filter_data: Any = None,
        num_workers: int = 0,
        filter_year: int | None = None,
    ):
        # Ensure we have the frames even in warm start
        if self.node_features is None and filter_data is not None:
             # In warm start, the pipeline passes filter_data (year) but we might need features
             # For the benchmark, we assume the caller ensures backend has frames
             pass
        from ..samplers.caracal_sampler import CaracalSampler
        sampler = CaracalSampler(
            self._get_db(),
            self.node_features,
            self.node_labels,
            fanouts,
            batch_size,
            scenario=scenario,
            filter_data=filter_data,
            num_workers=num_workers,
            filter_year=filter_year if filter_year is not None else self.config.filter_year,
        )
        return sampler

    def fetch_features(self, node_ids: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        if hasattr(self.node_features, "to_tensor"):
            try:
                # Try positional arguments for our locally patched Rust extension
                return self.node_features.to_tensor(None, node_ids, "float32", None, True, out)
            except TypeError:
                # Fallback to keyword arguments (will error if out is provided but not supported)
                if out is not None:
                    raise
                return self.node_features.to_tensor(indices=node_ids, dtype="float32")
        from ..ingest import gather_df_rows, convert_to_tensor
        indices = node_ids.tolist()
        batch_feat_df = gather_df_rows(self.node_features, indices)
        return convert_to_tensor(batch_feat_df)

    def close(self):
        if self._db:
            self._db.close()
            self._db = None
