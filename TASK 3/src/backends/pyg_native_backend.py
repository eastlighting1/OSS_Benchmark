import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from typing import Any, List
from .base import BaseBackend
from ..config import BenchmarkConfig
from ..ingest import convert_to_tensor, get_edge_index

class PyGNativeBackend(BaseBackend):
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.data = None

    def ingest(self, node_feat_df, node_label_df, node_year_df, edge_df):
        print("Ingesting data into PyG Native (In-memory)...")
        x = convert_to_tensor(node_feat_df)
        y = convert_to_tensor(node_label_df)
        edge_index = get_edge_index(edge_df)
        
        self.data = Data(x=x, y=y, edge_index=edge_index)

    def get_sampler(
        self,
        fanouts: List[int],
        batch_size: int,
        scenario: str = "default",
        filter_data: Any = None,
        num_workers: int = 0,
        filter_year: int | None = None,
    ):
        if self.data is None:
            raise ValueError("Data not ingested. Call ingest() first.")
        
        # Use our robust, dependency-free sampler instead of NeighborLoader
        return PureTorchNeighborSampler(
            self.data,
            fanouts,
            batch_size,
            scenario=scenario,
            filter_data=filter_data,
            filter_year=filter_year if filter_year is not None else self.config.filter_year,
        )
        
    def fetch_features(self, node_ids: torch.Tensor) -> torch.Tensor:
        return self.data.x[node_ids.long()]

class PureTorchNeighborSampler:
    def __init__(self, data, fanouts, batch_size, scenario="default", filter_data=None, filter_year=2014):
        self.data = data
        self.fanouts = fanouts
        self.batch_size = batch_size
        self.total_nodes = data.x.size(0)
        self.indices = np.arange(self.total_nodes)
        self.scenario = scenario
        self.filter_data = filter_data
        self.filter_year = filter_year
        
        # Scenario: Filtered (Filter SEEDS, not the graph space)
        if self.scenario == "filtered" and self.filter_data is not None:
            from ..ingest import convert_to_tensor
            years = convert_to_tensor(self.filter_data).flatten()
            mask = (years >= self.filter_year).numpy()
            self.indices = self.indices[mask[:len(self.indices)]]
            print(f"PyG Native: Filtered seeds to {len(self.indices)} nodes.")

        np.random.shuffle(self.indices)
        
        # Build adjacency list using FULL address space
        self.adj = [[] for _ in range(self.total_nodes)]
        edge_index = data.edge_index.numpy()
        for i in range(edge_index.shape[1]):
            src, dst = int(edge_index[0, i]), int(edge_index[1, i])
            if src < self.total_nodes and dst < self.total_nodes:
                self.adj[src].append(dst)

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.indices):
            raise StopIteration
        
        import time
        start_time = time.time()
        
        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        self.current_idx += self.batch_size
        
        all_nodes = set(int(node) for node in batch_indices)
        edge_list = []
        
        current_layer = [int(node) for node in batch_indices]
        for fanout in self.fanouts:
            next_layer = []
            for node in current_layer:
                neighbors = self.adj[node]
                if neighbors:
                    sampled = np.random.choice(neighbors, min(len(neighbors), fanout), replace=False)
                    for s in sampled:
                        s = int(s)
                        edge_list.append([node, s])
                        if s not in all_nodes:
                            all_nodes.add(s)
                        next_layer.append(s)
            current_layer = next_layer

        seeds = [int(node) for node in batch_indices]
        node_list = seeds + [node for node in all_nodes if node not in seeds]
        node_map = {node_id: i for i, node_id in enumerate(node_list)}
        
        local_edges = [[node_map[s], node_map[d]] for s, d in edge_list]
        edge_index = torch.tensor(local_edges, dtype=torch.long).t() if local_edges else torch.zeros((2, 0), dtype=torch.long)
        
        x = self.data.x[node_list]
        y = self.data.y[batch_indices]
        
        return x, y, edge_index, time.time() - start_time
