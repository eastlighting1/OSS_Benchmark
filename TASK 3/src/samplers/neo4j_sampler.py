import torch
import numpy as np
import time

class Neo4jSampler:
    def __init__(self, driver, node_features, fanouts, batch_size, scenario="default", filter_data=None):
        from ..ingest import get_df_len
        self.driver = driver
        self.node_features = node_features
        self.fanouts = fanouts
        self.batch_size = batch_size
        self.scenario = scenario
        self.filter_data = filter_data
        self.total_nodes = get_df_len(node_features)
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
        
        seeds = [str(int(x)) for x in batch_indices]
        
        # STRICT 2-HOP Cypher (Real Load)
        all_nodes = set(seeds)
        edge_list = []
        
        with self.driver.session() as session:
            # Multi-hop sampling via Cypher (returns edges explicitly)
            result = session.run("""
                UNWIND $seeds AS s
                MATCH p=(start:Node {node_id: s})-[:CITES*1..2]->(neighbor)
                WITH p, relationships(p) AS rels
                UNWIND rels AS rel
                RETURN startNode(rel).node_id AS src, endNode(rel).node_id AS dst
                LIMIT $limit
            """, seeds=seeds, limit=len(seeds) * (self.fanouts[0] + self.fanouts[1]))

            for record in result:
                src, dst = record["src"], record["dst"]
                edge_list.append([src, dst])
                all_nodes.add(dst)

        # Feature Fetching via universal gather_df_rows
        from ..ingest import gather_df_rows, convert_to_tensor, ensure_list
        node_list = list(all_nodes)
        node_indices = [int(n) for n in node_list if n.isdigit() and int(n) < self.total_nodes]
        batch_feat_df = gather_df_rows(self.node_features, node_indices)
        x = convert_to_tensor(batch_feat_df)
        
        y = torch.zeros(len(batch_indices))
        
        # Build local edge index
        node_map = {node_id: i for i, node_id in enumerate(node_list)}
        local_edges = [[node_map[s], node_map[d]] for s, d in edge_list if s in node_map and d in node_map]
        edge_index = torch.tensor(local_edges, dtype=torch.long).t() if local_edges else torch.zeros((2, 0), dtype=torch.long)
        
        return x, y, edge_index, time.time() - start_time
