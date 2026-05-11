import os
from neo4j import GraphDatabase
import torch
from typing import List
from .base import BaseBackend
from ..config import BenchmarkConfig

class Neo4jBackend(BaseBackend):
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.uri = os.environ.get("NEO4J_URI")
        self.user = os.environ.get("NEO4J_USER")
        self.password = os.environ.get("NEO4J_PASSWORD")
        
        if not all([self.uri, self.user, self.password]):
            raise ValueError("Neo4j configuration missing! Set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD environment variables.")
            
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.node_features = None

    def ingest(self, node_feat_df, node_label_df, node_year_df, edge_df):
        from ..ingest import get_df_len, get_df_cols, ensure_list
        print(f"Ingesting data into REAL Neo4j at {self.uri} (High-speed mode)...")
        self.node_features = node_feat_df
        
        with self.driver.session() as session:
            # 1. Clear DB and Create Index (CRITICAL for speed)
            session.run("MATCH (n) DETACH DELETE n")
            session.run("CREATE CONSTRAINT node_id_idx IF NOT EXISTS FOR (n:Node) REQUIRE n.node_id IS UNIQUE")
            
            # 2. Ingest Nodes in Chunks
            num_nodes = get_df_len(node_feat_df)
            node_data = [{"id": str(i)} for i in range(num_nodes)]
            
            # Chunking helper
            def chunker(seq, size):
                for pos in range(0, len(seq), size):
                    return seq[pos:pos + size] # Oops, fixed below

            # Correct Chunking and Ingestion
            batch_size = 5000
            for i in range(0, len(node_data), batch_size):
                batch = node_data[i:i + batch_size]
                session.run("UNWIND $data AS row CREATE (n:Node {node_id: row.id})", data=batch)
            
            # 3. Ingest Edges in Chunks
            import pandas as pd
            if hasattr(edge_df, "to_arrow"):
                pdf = edge_df.to_arrow().to_pandas()
            elif hasattr(edge_df, "compute"):
                pdf = edge_df.compute()
            else:
                pdf = edge_df
                
            cols = pdf.columns.tolist()
            data_cols = [c for c in cols if not str(c).startswith("_")]
            edge_list = pdf[data_cols].values.tolist()
            
            edge_data = [{"src": str(int(s)), "dst": str(int(d))} for s, d in edge_list]
            
            for i in range(0, len(edge_data), batch_size):
                batch = edge_data[i:i + batch_size]
                session.run("""
                    UNWIND $data AS row
                    MATCH (a:Node {node_id: row.src}), (b:Node {node_id: row.dst})
                    CREATE (a)-[:CITES]->(b)
                """, data=batch)
        print("Neo4j Ingestion complete.")

    def get_sampler(self, fanouts: List[int], batch_size: int, scenario: str = "default", filter_data: Any = None, num_workers: int = 0):
        from ..samplers.neo4j_sampler import Neo4jSampler
        return Neo4jSampler(self.driver, self.node_features, fanouts, batch_size, scenario=scenario, filter_data=filter_data)

    def fetch_features(self, node_ids: torch.Tensor) -> torch.Tensor:
        from ..ingest import gather_df_rows, convert_to_tensor
        indices = node_ids.tolist()
        batch_feat_df = gather_df_rows(self.node_features, indices)
        return convert_to_tensor(batch_feat_df)

    def close(self):
        if self.driver:
            self.driver.close()
