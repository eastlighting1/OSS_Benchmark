import torch
import numpy as np
import time

class DuckDBSampler:
    def __init__(self, conn, node_features, node_labels, fanouts, batch_size, scenario="default", filter_data=None, filter_year=2014):
        from ..ingest import get_df_len
        self.conn = conn
        self.node_features = node_features
        self.node_labels = node_labels
        self.fanouts = fanouts
        self.batch_size = batch_size
        self.scenario = scenario
        self.filter_data = filter_data
        self.filter_year = filter_year
        self.total_nodes = get_df_len(node_features)
        self.num_nodes = self.total_nodes 
        self.indices = np.arange(self.total_nodes)
        
        # Scenario: Filtered
        if self.scenario == "filtered" and self.filter_data is not None:
            from ..ingest import convert_to_tensor
            years = convert_to_tensor(self.filter_data).flatten()
            mask = (years >= self.filter_year).numpy()
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
        
        all_nodes = set(seeds)
        edge_list = []

        current_layer = seeds
        for hop, fanout in enumerate(self.fanouts):
            if len(current_layer) == 0:
                break

            frontier = ",".join(map(str, (int(x) for x in current_layer)))
            # Match CaracalDB/PyG semantics: sample up to `fanout` edges per
            # source node, not a global LIMIT across the whole frontier.
            query = f"""
                SELECT src, dst
                FROM (
                    SELECT
                        src,
                        dst,
                        row_number() OVER (
                            PARTITION BY src
                            ORDER BY hash(src, dst, {hop})
                        ) AS rn
                    FROM edges
                    WHERE src IN ({frontier})
                )
                WHERE rn <= {int(fanout)}
            """
            res = self.conn.execute(query).fetchall()
            if not res:
                break

            next_layer = []
            for src, dst in res:
                src = int(src)
                dst = int(dst)
                edge_list.append([src, dst])
                if dst not in all_nodes:
                    all_nodes.add(dst)
                next_layer.append(dst)
            current_layer = np.unique(np.asarray(next_layer, dtype=np.int64))

        # Keep seed nodes first, matching the CaracalDB loader contract used
        # by the benchmark label slicing.
        node_list = seeds + [node for node in all_nodes if node not in seeds]
        node_indices = [int(n) for n in node_list if n < self.total_nodes]
        
        from ..ingest import gather_df_rows, convert_to_tensor
        batch_feat_df = gather_df_rows(self.node_features, node_indices)
        x = convert_to_tensor(batch_feat_df)
        batch_label_df = gather_df_rows(self.node_labels, seeds)
        y = convert_to_tensor(batch_label_df)
        
        # Build local edge index
        node_map = {node_id: i for i, node_id in enumerate(node_list)}
        local_edges = [[node_map[s], node_map[d]] for s, d in edge_list if s in node_map and d in node_map]
        edge_index = torch.tensor(local_edges, dtype=torch.long).t() if local_edges else torch.zeros((2, 0), dtype=torch.long)
        
        return x, y, edge_index, time.time() - start_time
