from __future__ import annotations
from abc import ABC, abstractmethod
import time
import torch
from typing import List, Dict, Any

class BaseBackend(ABC):
    @abstractmethod
    def ingest(self, node_feat_df, node_label_df, node_year_df, edge_df):
        """
        Store data in the backend.
        """
        pass

    @abstractmethod
    def get_sampler(self, fanouts: List[int], batch_size: int, scenario: str = "default", filter_data: Any = None, num_workers: int = 0):
        """
        Return a sampler/loader for GNN training with scenario support.
        """
        pass

    @abstractmethod
    def fetch_features(self, node_ids: torch.Tensor) -> torch.Tensor:
        """
        Fetch features for a batch of node IDs.
        """
        pass

import os
import psutil
import torch

class BenchmarkMetrics:
    def __init__(self, start_time: float = None):
        self.ttfb = 0.0
        self.edges_per_sec = 0.0
        self.data_wait_ratio = 0.0
        self.memory_peak_mb = 0.0
        self.gpu_memory_peak_mb = 0.0
        self.epoch_times = []
        self.start_time = start_time if start_time else time.time()
        self.process = psutil.Process(os.getpid())

    def start_timer(self):
        # Only set if not already set by external logic (like Ingest)
        if not self.start_time:
            self.start_time = time.time()
        self.update_memory()

    def record_ttfb(self):
        self.ttfb = time.time() - self.start_time
        self.update_memory()

    def update_memory(self):
        # RAM Tracking (RSS)
        mem_info = self.process.memory_info()
        current_ram = mem_info.rss / (1024 * 1024)
        if current_ram > self.memory_peak_mb:
            self.memory_peak_mb = current_ram
            
        # GPU Tracking
        if torch.cuda.is_available():
            current_gpu = torch.cuda.max_memory_allocated() / (1024 * 1024)
            if current_gpu > self.gpu_memory_peak_mb:
                self.gpu_memory_peak_mb = current_gpu

    def report(self):
        return {
            "ttfb": self.ttfb,
            "edges_per_sec": self.edges_per_sec,
            "data_wait_ratio": self.data_wait_ratio,
            "memory_peak_mb": self.memory_peak_mb,
            "gpu_peak_mb": self.gpu_memory_peak_mb
        }
