from __future__ import annotations
from pathlib import Path
import lynxes
import torch
import numpy as np
from .config import BenchmarkConfig

import pyarrow.csv as pv
import pyarrow as pa

def create_lynxes_node_frame(table, label):
    n = table.num_rows
    table = table.append_column("_id", pa.array([str(i) for i in range(n)], type=pa.string()))
    table = table.append_column(
        "_label",
        pa.array([[label] for _ in range(n)], type=pa.list_(pa.string())),
    )
    columns = ["_id", "_label", *[name for name in table.column_names if not name.startswith("_")]]
    selected = table.select(columns).combine_chunks()
    batches = selected.to_batches(max_chunksize=max(1, selected.num_rows))
    if batches:
        batch = batches[0]
    else:
        batch = pa.RecordBatch.from_arrays(
            [pa.array([], type=field.type) for field in selected.schema],
            schema=selected.schema,
        )
    return lynxes.NodeFrame.from_arrow(batch)

def load_gnn_data(config: BenchmarkConfig):
    sample_n = config.sample_nodes
    print(f"Loading node features (sampling {sample_n} nodes)")
    feat_cols = [str(i) for i in range(128)]
    node_feat_table = pv.read_csv(config.node_feat_path, 
                                  read_options=pv.ReadOptions(column_names=feat_cols),
                                  convert_options=pv.ConvertOptions(include_columns=feat_cols)).slice(0, sample_n)
    node_feat_df = create_lynxes_node_frame(node_feat_table, "Node")
    
    print(f"Loading node labels")
    node_label_table = pv.read_csv(config.node_label_path, read_options=pv.ReadOptions(column_names=["label"])).slice(0, sample_n)
    node_label_df = create_lynxes_node_frame(node_label_table, "Node")
    
    print(f"Loading node years")
    node_year_table = pv.read_csv(config.node_year_path, read_options=pv.ReadOptions(column_names=["year"])).slice(0, sample_n)
    node_year_df = create_lynxes_node_frame(node_year_table, "Node")
    
    print(f"Loading edges (Standardized)")
    edge_table = pv.read_csv(config.edge_path, read_options=pv.ReadOptions(column_names=["src", "dst"]))
    
    import pyarrow.compute as pc
    mask = pc.and_(pc.less(edge_table["src"], sample_n), pc.less(edge_table["dst"], sample_n))
    edge_table = edge_table.filter(mask)
    edge_df = create_lynxes_node_frame(edge_table, "Edge")
    
    return node_feat_df, node_label_df, node_year_df, edge_df

def process_and_align_data(node_feat_df, node_label_df, node_year_df, edge_df):
    return node_feat_df, node_label_df, node_year_df, edge_df

def ensure_list(obj):
    if isinstance(obj, list): return obj
    if hasattr(obj, "tolist"): return obj.tolist()
    return list(obj)

def get_df_len(df):
    if hasattr(df, "len") and callable(df.len): return df.len()
    try: return len(df)
    except: return 0

def get_df_cols(df):
    if hasattr(df, "column_names"):
        res = df.column_names() if callable(df.column_names) else df.column_names
        return list(res)
    if hasattr(df, "columns"):
        return list(df.columns)
    return []

def gather_df_rows(df, indices):
    indices = ensure_list(indices)
    # THIN DELEGATION: Favor library methods
    if hasattr(df, "to_arrow"):
        return df.to_arrow().take(indices)
    
    import dask.dataframe as dd
    if isinstance(df, dd.DataFrame):
        return df.compute().iloc[indices]
    
    if hasattr(df, "iloc"):
        return df.iloc[indices]
    
    return df[indices]

def convert_to_tensor(df, column_names=None):
    # THIN DELEGATION: Favor library native tensor conversion
    if df.__class__.__module__.split(".", 1)[0] == "lynxes" and hasattr(df, "to_tensor"):
        kwargs = {"dtype": "float32"}
        if column_names is not None:
            kwargs["columns"] = column_names
        return df.to_tensor(**kwargs)

    if hasattr(df, "to_torch_tensor"):
        return df.to_torch_tensor(column_names)

    import pandas as pd
    is_arrow = isinstance(df, (pa.Table, pa.RecordBatch))
    arrow_obj = df if is_arrow else (df.to_arrow() if hasattr(df, "to_arrow") else None)
    
    if arrow_obj is not None:
        try:
            cols = arrow_obj.column_names if hasattr(arrow_obj, "column_names") else arrow_obj.schema.names
            data_cols = [c for c in cols if not str(c).startswith("_")]
            target_cols = column_names if column_names else data_cols
            arrays = [arrow_obj.column(c).to_numpy() for c in target_cols]
            stacked = np.column_stack(arrays)
            return torch.from_numpy(stacked.copy()).float()
        except: pass

    try:
        pdf = df.compute() if hasattr(df, "compute") else (df.to_pandas() if hasattr(df, "to_pandas") else df)
        if isinstance(pdf, pd.DataFrame):
            data_cols = [c for c in pdf.columns if not str(c).startswith("_")]
            target_cols = column_names if column_names else data_cols
            return torch.from_numpy(pdf[target_cols].values.copy()).float()
    except: pass
    return torch.tensor([])

def get_edge_index(edge_df):
    cols = get_df_cols(edge_df)
    data_cols = [c for c in cols if not str(c).startswith("_")]
    src_col, dst_col = data_cols[0], data_cols[1]
    
    if hasattr(edge_df, "to_arrow"):
        table = edge_df.to_arrow()
        src = table.column(src_col).to_numpy()
        dst = table.column(dst_col).to_numpy()
        edge_index = np.vstack([src, dst])
    else:
        pdf = edge_df.compute() if hasattr(edge_df, "compute") else (edge_df.to_pandas() if hasattr(edge_df, "to_pandas") else edge_df)
        edge_index = pdf[[src_col, dst_col]].values.T
        
    return torch.from_numpy(edge_index.copy()).long()
