from __future__ import annotations

import time
from typing import Any, Iterable

import numpy as np
import torch


class CaracalSampler:
    """Thin benchmark wrapper around CaracalDB's official GNN loader API."""

    def __init__(
        self,
        db: Any,
        node_features: Any,
        node_labels: Any,
        fanouts: Iterable[int],
        batch_size: int,
        scenario: str = "default",
        filter_data: Any = None,
        num_workers: int = 0,
        filter_year: int = 2014,
    ):
        self.db = db
        self.node_features = node_features
        self.node_labels = node_labels
        self.fanouts = tuple(int(fanout) for fanout in fanouts)
        self.batch_size = int(batch_size)
        self.scenario = scenario
        self.filter_data = filter_data
        self.num_workers = int(num_workers)
        self.filter_year = int(filter_year)
        self._loader_iter = None
        self._feat_buffer = None
        self._label_buffer = None

    def __iter__(self):
        input_nodes = self._scenario_input()
        kwargs = dict(
            batch_size=self.batch_size,
            shuffle=True,
            filter=None,
            warm_start=self.scenario == "warm_start",
            num_workers=self.num_workers,
            seed=0,
            return_format="pyg",
        )
        try:
            loader = self.db.neighbor_loader(
                input_nodes,
                self.fanouts,
                ("CITES",),
                **kwargs,
            )
        except TypeError:
            kwargs.pop("seed")
            loader = self.db.neighbor_loader(
                input_nodes,
                self.fanouts,
                ("CITES",),
                **kwargs,
            )
        self._loader_iter = iter(loader)
        return self

    def __next__(self):
        if self._loader_iter is None:
            self.__iter__()

        start = time.time()
        edge_index_np, n_id_np = next(self._loader_iter)
        n_id = np.asarray(n_id_np, dtype=np.int64)

        seed_nodes = n_id[: min(self.batch_size, len(n_id))]
        x = self._frame_to_tensor(self.node_features, n_id, is_feature=True)
        y = self._frame_to_tensor(self.node_labels, seed_nodes, is_feature=False)
        edge_index = torch.from_numpy(np.asarray(edge_index_np)).long()

        if y.dim() == 0:
            y = y.unsqueeze(0)

        return x, y, edge_index, time.time() - start

    def _scenario_input(self) -> np.ndarray | None:
        if self.scenario != "filtered":
            return None

        query = f"MATCH (n:Node) WHERE n.year >= {self.filter_year} RETURN n.node_id AS node_id"
        table = self.db.sql(query).arrow()
        return table.column("node_id").to_numpy().astype(np.int64)

    def _frame_to_tensor(self, frame: Any, indices: np.ndarray | None, is_feature: bool = True) -> torch.Tensor:
        if _is_lynxes_frame(frame) and hasattr(frame, "to_tensor"):
            # Use pre-allocated buffer
            buffer = self._feat_buffer if is_feature else self._label_buffer
            num_rows = len(indices) if indices is not None else len(frame)
            
            if buffer is None or buffer.shape[0] < num_rows:
                # Need to allocate new buffer
                try:
                    out_tensor = frame.to_tensor(None, indices, "float32", None, True, None)
                except TypeError:
                    out_tensor = frame.to_tensor(indices=indices, dtype="float32")
                
                if is_feature:
                    self._feat_buffer = out_tensor
                else:
                    self._label_buffer = out_tensor
                return out_tensor
            else:
                # Reuse buffer (slice to exact size)
                out_view = buffer[:num_rows]
                try:
                    return frame.to_tensor(None, indices, "float32", None, True, out_view)
                except TypeError:
                    return frame.to_tensor(indices=indices, dtype="float32")

        from ..ingest import convert_to_tensor, gather_df_rows

        if indices is not None:
            frame = gather_df_rows(frame, indices.tolist())
        return convert_to_tensor(frame)


def _is_lynxes_frame(frame: Any) -> bool:
    return frame.__class__.__module__.split(".", 1)[0] == "lynxes"
