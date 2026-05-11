import torch
import numpy as np
import time

class DuckDBSampler:
    def __init__(self, conn, node_features, fanouts, batch_size, scenario="default", filter_data=None):
        from ..ingest import get_df_len
        self.conn = conn
        self.node_features = node_features
        self.fanouts = fanouts
        self.batch_size = batch_size
        self.scenario = scenario
        self.filter_data = filter_data
        self.total_nodes = get_df_len(node_features)
        self.num_nodes = self.total_nodes 
        self.indices = np.arange(self.total_nodes)
        
        # Scenario: Filtered
        if self.scenario == "filtered" and self.filter_data is not None:
            from ..ingest import convert_to_tensor
            years = convert_to_tensor(self.filter_data).flatten()
            threshold = float(torch.median(years))
            mask = (years >= threshold).numpy()
            self.indices = self.indices[mask[:len(self.indices)]]

        np.random.shuffle(self.indices)

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
        
        seeds = [int(x) for x in batch_indices]
        seed_str = ",".join(map(str, seeds))
        
        # STRICT 2-HOP Recursive JOIN (Real Load)
        query = f"""
            WITH hop1 AS (
                SELECT src, dst FROM edges WHERE src IN ({seed_str}) LIMIT {len(seeds) * self.fanouts[0]}
            ),
            hop2 AS (
                SELECT e.src, e.dst FROM edges e JOIN hop1 h ON e.src = h.dst LIMIT {len(seeds) * self.fanouts[1]}
            )
            SELECT src, dst FROM hop1 UNION SELECT src, dst FROM hop2
        """
        res = self.conn.execute(query).fetchall()
        
        all_nodes = set(seeds)
        edge_list = []
        for src, dst in res:
            edge_list.append([src, dst])
            all_nodes.add(dst)

        node_list = list(all_nodes)
        node_indices = [int(n) for n in node_list if n < self.total_nodes]
        
        from ..ingest import gather_df_rows, convert_to_tensor
        batch_feat_df = gather_df_rows(self.node_features, node_indices)
        x = convert_to_tensor(batch_feat_df)
        y = torch.zeros(len(seeds)) # Labels placeholder for benchmark
        
        # Build local edge index
        node_map = {node_id: i for i, node_id in enumerate(node_list)}
        local_edges = [[node_map[s], node_map[d]] for s, d in edge_list if s in node_map and d in node_map]
        edge_index = torch.tensor(local_edges, dtype=torch.long).t() if local_edges else torch.zeros((2, 0), dtype=torch.long)
        
        return x, y, edge_index, time.time() - start_time
