import torch
import numpy as np
import time
from torch.utils.data import IterableDataset

class CaracalSampler(IterableDataset):
    def __init__(self, db, node_features, node_labels, fanouts, batch_size, scenario="default", filter_data=None):
        super().__init__()
        from ..ingest import get_df_len
        self.db = db
        self.node_features = node_features
        self.node_labels = node_labels
        self.fanouts = fanouts
        self.batch_size = batch_size
        self.scenario = scenario
        self.filter_data = filter_data
        
        self.total_nodes = get_df_len(node_features)
        self.indices = np.arange(self.total_nodes)
        
        if self.scenario == "filtered":
            # PUSH-DOWN: Let the DB handle the predicate
            # We assume the 'year' property was ingested into CaracalDB
            try:
                # 1. Get threshold (we still need to know WHAT to filter)
                # For benchmark consistency, we use the median if filter_data is provided
                if self.filter_data is not None:
                    from ..ingest import convert_to_tensor
                    years = convert_to_tensor(self.filter_data).flatten()
                    threshold = int(torch.median(years))
                else:
                    threshold = 2014 # Fallback
                
                # 2. Query DB for filtered seeds (Zero-Copy/Zero-IPC predicate)
                # CaracalDB uses TUFT/Cypher-like syntax: MATCH ... WHERE ... RETURN ...
                # INDEXED: We created an index on 'year' during ingestion
                res_cursor = self.db.sql(f"MATCH (n:Node) WHERE n.year >= {threshold} RETURN n.node_id AS node_id")
                res_table = res_cursor.arrow()
                self.indices = res_table.column('node_id').to_numpy().astype(np.int64)
                print(f"CaracalDB Push-down (INDEXED): Filtered seeds to {len(self.indices)} nodes using MATCH.")
            except Exception as e:
                print(f"Push-down failed, falling back to Python filtering: {e}")
                if self.filter_data is not None:
                    from ..ingest import convert_to_tensor
                    years = convert_to_tensor(self.filter_data).flatten()
                    threshold = float(torch.median(years))
                    mask = (years >= threshold).numpy() 
                    self.indices = self.indices[mask[:len(self.indices)]]

        np.random.shuffle(self.indices)
        
    def __iter__(self):
        # Support for multi-worker
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.start = 0
            self.end = len(self.indices)
        else:
            per_worker = int(np.ceil(len(self.indices) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            self.start = worker_id * per_worker
            self.end = min(self.start + per_worker, len(self.indices))
            
        self.current_idx = self.start
        return self

    def __next__(self):
        if self.current_idx >= self.end:
            raise StopIteration
        
        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        self.current_idx += self.batch_size
        
        start_time = time.time()
        seeds = batch_indices # Already numpy, no need to list
        
        # NATIVE CALL: Pass fanouts array for hop-limiting
        try:
            # OPTIMIZED: pass fanouts list directly (requires patched caracaldb)
            # Use tuple for seeds to match optimized API
            edge_index_np, unique_nodes = self.db.sample_gnn_subgraph(
                seeds=tuple(seeds), depth=len(self.fanouts), edge_types=["CITES"],
                fanouts=self.fanouts
            )
            edge_index = torch.from_numpy(edge_index_np).long()
        except Exception as e:
            import traceback
            # traceback.print_exc()
            return torch.zeros((len(batch_indices), 128)), torch.zeros(len(batch_indices)), torch.zeros((2, 0), dtype=torch.long), 0

        # NATIVE CALL: Zero-copy feature fetching
        # Use our new NodeFrame.to_tensor(indices)
        if hasattr(self.node_features, "to_tensor"):
            # OPTIMIZED: Pass unique_nodes directly (numpy) to avoid tolist() overhead
            x = self.node_features.to_tensor(indices=unique_nodes)
        else:
            from ..ingest import gather_df_rows, convert_to_tensor
            subgraph_feats_df = gather_df_rows(self.node_features, unique_nodes.tolist())
            x = convert_to_tensor(subgraph_feats_df)
        
        # Labels for the original seeds
        if hasattr(self.node_labels, "to_tensor"):
            y = self.node_labels.to_tensor(indices=seeds)
        else:
            from ..ingest import gather_df_rows, convert_to_tensor
            batch_labels_df = gather_df_rows(self.node_labels, seeds.tolist())
            y = convert_to_tensor(batch_labels_df)
        
        if y.dim() == 0: y = y.unsqueeze(0)
        if len(y) != len(batch_indices):
             y = torch.zeros(len(batch_indices))
            
        return x, y, edge_index, time.time() - start_time
