from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT.parent / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

@dataclass
class BenchmarkConfig:
    dataset: str = "ogbn-arxiv"
    data_dir: Path = DATA_DIR
    output_dir: Path = OUTPUT_DIR
    sample_nodes: int = 10000 # Default sample size
    
    # Scenario Params
    scenario: str = "default"
    filter_year: int = 2014
    num_workers: int = 0
    
    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
    
    # GNN Params
    batch_size: int = 1024
    fanouts: List[int] = field(default_factory=lambda: [15, 10])
    epochs: int = 5
    lr: float = 0.01
    hidden_channels: int = 256
    num_layers: int = 2
    
    # Backend Params
    caracal_db_name: str = "gnn_benchmark"
    
    @property
    def node_feat_path(self) -> Path:
        return self.data_dir / "node-feat.csv"
    
    @property
    def node_label_path(self) -> Path:
        return self.data_dir / "node-label.csv"
    
    @property
    def node_year_path(self) -> Path:
        return self.data_dir / "node_year.csv"
    
    @property
    def edge_path(self) -> Path:
        return self.data_dir / "edge.csv"

def get_config(dataset: str = "ogbn-arxiv", **kwargs) -> BenchmarkConfig:
    return BenchmarkConfig(dataset=dataset, **kwargs)
