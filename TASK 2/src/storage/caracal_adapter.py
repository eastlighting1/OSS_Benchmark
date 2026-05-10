from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any
from collections import defaultdict

import lynxes as lx
import numpy as np
import pyarrow as pa

from ..models import Answer, Chunk, Citation, ContextItem, Entity, GraphArtifacts, QueryEntityLink, RetrievalPlan, Row, SemanticCandidate
from ..retrieval_strategy import STOPWORDS
from .base import NativeGraphRetrieval, StorageAdapter, link_query_entities_from_entities, merge_query_entity_links


class CaracalStorageAdapter(StorageAdapter):
    config_id = "caracal-only"
    config_name = "CaracalDB only"
    graph_db = "caracaldb"
    vector_db = "none"

    def __init__(self, db_path: Path):
        super().__init__()
        self.db_path = db_path
        self._db = None
        self.chunk_vector_index_name = "chunk_vector_idx"
        self.entity_vector_index_name = "entity_vector_idx"
        self.capabilities: dict[str, Any] = {}
        self.native_vector_ready = False
        self.native_neighbors_ready = False
        self.native_paths_ready = False
        self.native_text_index_ready = False
        self.native_entity_vector_ready = False
        self.native_graphrag_ready = False
        self._cached_filtered_entities = None

    def load(self, artifacts: GraphArtifacts) -> None:
        """Loads artifacts into CaracalDB and performs graph-native optimizations."""
        super().load(artifacts)
        db = self._get_db()
        try:
            self.capabilities = discover_capabilities(db)
            self.sync(db)
            
            # 1. Bulk load Nodes using Arrow
            tables = artifacts_to_tables(artifacts)
            for table_name, table_data in zip(["Document", "Chunk", "Entity", "Embedding"], tables):
                if hasattr(db, "insert_node_table_arrow"):
                    db.insert_node_table_arrow(
                        rows_to_arrow(table_data), 
                        key_col="node_id" if table_name != "Embedding" else "owner_id"
                    )
            
            # 2. Bulk load Edges using Arrow
            if hasattr(db, "insert_edge_table_arrow"):
                db.insert_edge_table_arrow(rows_to_arrow(edge_rows(artifacts)))

            # 3. Create Vector Indexes
            if self.native_vector_ready:
                # Detect dimension from data
                dim = len(artifacts.embeddings[0].vector) if artifacts.embeddings else 256
                
                db.create_vector_index(
                    name=self.chunk_vector_index_name,
                    node_type="Chunk",
                    property="embedding",
                    dimension=dim,
                    metric="cosine"
                )
                db.create_vector_index(
                    name=self.entity_vector_index_name,
                    node_type="Entity",
                    property="embedding",
                    dimension=dim,
                    metric="cosine"
                )

            # 4. Analytics & Smoothing
            self._compute_and_store_graph_analytics(db, artifacts)
            
        finally:
            if hasattr(db, "close") and callable(db.close):
                db.close()

    def _compute_and_store_graph_analytics(self, db: Any, artifacts: GraphArtifacts) -> None:
        """Computes PageRank and Communities using Lynxes and stores them in CaracalDB."""
        if not artifacts:
            return
            
        try:
            # Build RecordBatches for Lynxes
            node_data = []
            id_to_type = {}
            known_node_ids = set()

            for entity_obj in artifacts.entities:
                node_data.append({"_id": entity_obj.entity_id, "_label": [entity_obj.entity_type]})
                id_to_type[entity_obj.entity_id] = "Entity"
                known_node_ids.add(entity_obj.entity_id)
            for chunk_obj in artifacts.chunks:
                node_data.append({"_id": chunk_obj.chunk_id, "_label": ["Chunk"]})
                id_to_type[chunk_obj.chunk_id] = "Chunk"
                known_node_ids.add(chunk_obj.chunk_id)
                
            e_rows = edge_rows(artifacts)
            edge_data = []
            dangling_node_ids = set()
            for edge in e_rows:
                src, dst = edge["src"], edge["dst"]
                edge_data.append({
                    "_src": src, 
                    "_dst": dst, 
                    "_type": edge.get("type", "RELATED_TO"),
                    "weight": edge.get("weight", 1.0)
                })
                if src not in known_node_ids: dangling_node_ids.add(src)
                if dst not in known_node_ids: dangling_node_ids.add(dst)
            
            for missing_id in dangling_node_ids:
                label = "Document" if missing_id.startswith("doc:") else "Unknown"
                node_data.append({"_id": missing_id, "_label": [label]})
                id_to_type[missing_id] = label
                
            if not node_data or not edge_data:
                return

            nodes_batch = pa.Table.from_pylist(node_data).to_batches()[0]
            edges_table = pa.Table.from_pylist(edge_data)
            edges_table = edges_table.append_column("_direction", pa.array([0] * len(edge_data), type=pa.int8()))
            edges_batch = edges_table.to_batches()[0]
            
            nf = lx.NodeFrame.from_arrow(nodes_batch)
            ef = lx.EdgeFrame.from_arrow(edges_batch)
            gf = nf.with_edges(ef)
            
            pr_results = gf.pagerank()
            pagerank_map = {row["_id"]: row["pagerank"] for row in pr_results.to_pyarrow().to_pylist()}
            max_pr = max(pagerank_map.values()) if pagerank_map else 1.0
            
            community_results = gf.community_detection()
            community_map = {row["_id"]: row["community_id"] for row in community_results.to_pyarrow().to_pylist()}
            num_communities = len(set(community_map.values()))
                    
            # Embedding Smoothing
            id_to_vec = {record.owner_id: record.vector for record in artifacts.embeddings}
            mentions_by_chunk = defaultdict(list)
            for m in artifacts.mentions:
                mentions_by_chunk[m.chunk_id].append(m.entity_id)

            smoothing_updates = []
            for chunk in artifacts.chunks:
                mentioned = mentions_by_chunk.get(chunk.chunk_id, [])
                if mentioned:
                    valid_entities = [(eid, pagerank_map.get(eid, 0.0)) for eid in mentioned if eid in id_to_vec]
                    if valid_entities:
                        chunk_vec = np.array(id_to_vec[chunk.chunk_id])
                        total_pr = sum(pr for _, pr in valid_entities) or 1.0
                        entity_weighted_avg = np.zeros_like(chunk_vec)
                        for eid, pr in valid_entities:
                            entity_weighted_avg += np.array(id_to_vec[eid]) * (pr / total_pr)
                        fused_vec = (0.6 * chunk_vec + 0.4 * entity_weighted_avg).tolist()
                        smoothing_updates.append({"node_id": chunk.chunk_id, "type": "Chunk", "embedding": fused_vec})
            
            # Analytical updates
            update_rows = []
            for node_id in id_to_type:
                pr_val = float(pagerank_map.get(node_id, 0.0)) / max_pr
                update_rows.append({
                    "node_id": node_id, "type": id_to_type[node_id],
                    "pagerank": pr_val, "community": int(community_map.get(node_id, -1))
                })

            if hasattr(db, "upsert_node_table_arrow"):
                db.upsert_node_table_arrow(rows_to_arrow(update_rows), key_col="node_id", update_existing=True)
                if smoothing_updates:
                    db.upsert_node_table_arrow(rows_to_arrow(smoothing_updates), key_col="node_id", update_existing=True)

            # Update memory objects
            self._cached_filtered_entities = [
                e for e in artifacts.entities 
                if (len(e.name) > 3 or e.name.isupper()) and e.name.lower() not in STOPWORDS and any(c.isalnum() for c in e.name)
            ]
            self.notes = append_note(self.notes, f"lynxes_analytics_completed: nodes={len(node_data)}, communities={num_communities}")
        except Exception as exc:
            self.notes = append_note(self.notes, f"lynxes_analytics_failed: {exc}")

    def semantic_entry(self, question: str, query_embedding: list[float], top_k: int):
        effective_top_k = top_k * 2 if self.config_id == "caracal-only" else top_k
        if self.native_vector_ready:
            base_candidates = self._try_native_vector_search(self.chunk_vector_index_name, query_embedding, effective_top_k, "Chunk")
            if self.config_id == "caracal-only":
                try:
                    db = self._get_db()
                    smart_seeds = self.link_query_entities(question, query_embedding, 10)
                    combined_seeds = list(dict.fromkeys([c.node_id for c in (base_candidates or [])] + [s.entity_id for s in smart_seeds]))
                    if combined_seeds:
                        pull_result = db.neighbors(seed_node_ids=combined_seeds, edge_types=["MENTIONS", "RELATED_TO", "HAS_CHUNK"], depth=1, limit=effective_top_k * 2, node_type_filters=["Chunk"])
                        pull_rows = result_to_rows(pull_result)
                        pulled = semantic_candidates_from_rows(pull_rows, source="caracal_deep_fusion_pull")
                        merged = {c.node_id: c for c in (base_candidates or [])}
                        for c in pulled:
                            if c.node_id not in merged: merged[c.node_id] = c
                        self.semantic_entry_mode = "caracal_hnsw_deep_fusion"
                        return sorted(merged.values(), key=lambda x: (-x.score, x.node_id))[:effective_top_k]
                except Exception as exc:
                    self.notes = append_note(self.notes, f"deep_fusion_pull_failed: {exc}")
            if base_candidates:
                self.semantic_entry_mode = "caracal_hnsw"
                return base_candidates
        return super().semantic_entry(question, query_embedding, top_k)

    def link_query_entities(self, question: str, query_embedding: list[float], top_k: int) -> list[QueryEntityLink]:
        if self.artifacts is not None:
            filtered_entities = getattr(self, "_cached_filtered_entities", None)
            if filtered_entities is None:
                filtered_entities = [e for e in self.artifacts.entities if (len(e.name) > 3 or e.name.isupper()) and e.name.lower() not in STOPWORDS and any(c.isalnum() for c in e.name)]
            lexical_links = link_query_entities_from_entities(question=question, entities=filtered_entities, embeddings_by_owner=self.embeddings_by_owner, query_embedding=query_embedding, top_k=top_k, source="caracal_smart_seed_lexical", use_vector_fallback=False)
            if lexical_links: return lexical_links
        return super().link_query_entities(question, query_embedding, top_k)

    def native_graph_retrieval(self, question: str, query_embedding: list[float], plan: RetrievalPlan) -> NativeGraphRetrieval | None:
        if not (self.native_graphrag_ready and self.native_vector_ready): return None
        if plan.strategy != "global_community_summary": return None
        try:
            db = self._get_db()
            result = db.graphrag_search(query_text=question, query_vector=query_embedding, chunk_vector_index=self.chunk_vector_index_name, edge_types=["MENTIONS", "RELATED_TO", "EVIDENCED_BY", "HAS_CHUNK"], max_depth=plan.relation_depth, semantic_top_k=48, entity_top_k=plan.entity_top_k, evidence_top_k=plan.evidence_budget, citation_top_k=plan.citation_budget, scoring={"semantic": 0.35, "entity_link": 0.50, "path": 0.25, "document_diversity": 0.1}, return_properties=["document_id", "text", "chunk_index", "pagerank", "community"], profile=True)
            self.semantic_entry_mode = "caracal_graphrag_search"
            self.relation_expand_mode = "caracal_graphrag_search"
            return NativeGraphRetrieval(semantic_candidates=semantic_candidates_from_rows(result_to_rows(result.semantic_hits)), query_entity_links=query_entity_links_from_rows(result_to_rows(result.entity_links)), context_items=context_items_from_graphrag_rows(result_to_rows(result.evidence_chunks)), profile=getattr(result, "profile", {}) or {}, operator_timings_ms={})
        except Exception as exc:
            self.notes = append_note(self.notes, f"native graphrag_search fallback: {exc}")
            return None

    def evidence_path_expand(self, semantic_candidates: list[SemanticCandidate], entity_links: list[QueryEntityLink], plan: RetrievalPlan) -> list[ContextItem]:
        if self.capabilities.get("graphrag.evidence_search"):
            native_evidence = self._try_native_evidence_search(semantic_candidates, entity_links, plan)
            if native_evidence is not None:
                self.relation_expand_mode = "caracal_evidence_search"
                return native_evidence
        return super().evidence_path_expand(semantic_candidates, entity_links, plan)

    def _get_db(self):
        if self._db is None:
            import caracaldb
            self._db = caracaldb.connect(self.db_path)
        return self._db

    def close(self):
        if self._db is not None:
            if hasattr(self._db, "close") and callable(self._db.close):
                self._db.close()
            self._db = None

    def sync(self, db: Any) -> None:
        self.native_vector_ready = bool(self.capabilities.get("vector_search"))
        self.native_neighbors_ready = bool(self.capabilities.get("traversal.neighbors"))
        self.native_paths_ready = bool(self.capabilities.get("traversal.paths"))
        self.native_text_index_ready = bool(self.capabilities.get("text_index"))
        self.native_entity_vector_ready = self.native_vector_ready
        self.native_graphrag_ready = bool(self.capabilities.get("graphrag.search"))

    def _try_native_vector_search(self, index_name: str, vector: list[float], top_k: int, node_type: str) -> list[SemanticCandidate]:
        try:
            db = self._get_db()
            result = db.vector_search(index=index_name, query_vector=vector, top_k=top_k, filters={"type": node_type}, return_properties=["document_id", "text", "chunk_index", "pagerank", "community"])
            return semantic_candidates_from_rows(result_to_rows(result), source="caracal_hnsw")
        except Exception as exc:
            self.notes = append_note(self.notes, f"native vector_search fallback: {exc}")
            return []

    def _try_native_evidence_search(self, semantic_candidates, entity_links, plan) -> list[ContextItem] | None:
        try:
            db = self._get_db()
            seed_scores = {c.node_id: c.score for c in semantic_candidates}
            for link in entity_links: seed_scores[link.entity_id] = max(seed_scores.get(link.entity_id, 0), link.score)

            # Use depth 3 to find other chunks: Seed(Chunk) -> Entity -> Entity -> Target(Chunk)
            # Removed 'document_id' to avoid schema mismatch issues if some nodes are not chunks
            result = db.evidence_search(
                seed_node_ids=list(seed_scores.keys()), 
                target_node_type="Chunk",
                edge_types=["MENTIONS", "RELATED_TO", "HAS_CHUNK"],
                direction="both",
                max_depth=3, 
                top_k=plan.evidence_budget, 
                scoring={"path_weight": 0.35, "seed_score": 0.25},
                return_properties=["text", "chunk_index", "pagerank", "community"], 
                seed_scores=seed_scores,
                node_key_col="node_id"
            )
            return context_items_from_graphrag_rows(result_to_rows(result))
        except Exception as exc:
            self.notes = append_note(self.notes, f"native evidence_search fallback: {exc}")
            return None



def discover_capabilities(db: Any) -> dict[str, Any]:
    return {
        "vector_search": hasattr(db, "vector_search"),
        "traversal.neighbors": hasattr(db, "neighbors"),
        "traversal.paths": hasattr(db, "paths"),
        "text_index": hasattr(db, "create_text_index") and hasattr(db, "text_search"),
        "graphrag.search": hasattr(db, "graphrag_search"),
        "graphrag.evidence_search": hasattr(db, "evidence_search"),
        "graphrag.link_entities": hasattr(db, "link_entities"),
    }

def semantic_candidates_from_rows(rows: list[Row], source: str = "caracal_graphrag_search") -> list[SemanticCandidate]:
    candidates = []
    for index, row in enumerate(rows):
        node_id = str(row.get("node_id") or row.get("chunk_id") or "")
        if not node_id: continue
        candidates.append(SemanticCandidate(node_id=node_id, node_type=str(row.get("node_type") or "Chunk"), score=float(row.get("score") or row.get("vector_score") or 0.0), rank=int(row.get("rank") or index + 1), reason="CaracalDB fused semantic entry", source=source))
    return candidates

def context_items_from_graphrag_rows(rows: list[Row]) -> list[ContextItem]:
    items = []
    for row in rows:
        node_id = str(row.get("node_id") or row.get("chunk_id") or "")
        if not node_id.startswith("chunk:"): continue
        items.append(ContextItem(node_id=node_id, node_type="Chunk", score=float(row.get("score") or 1.0), reason=str(row.get("reason") or "CaracalDB fused evidence"), path=row.get("path", [node_id])))
    return items

def query_entity_links_from_rows(rows: list[Row]) -> list[QueryEntityLink]:
    links = []
    for index, row in enumerate(rows):
        entity_id = str(row.get("node_id") or row.get("entity_id") or "")
        if not entity_id: continue
        links.append(QueryEntityLink(entity_id=entity_id, name=str(row.get("name") or ""), score=float(row.get("score") or 1.0), rank=index + 1, matched_text=str(row.get("matched_text") or ""), source="caracal_native_linker"))
    return links

def artifacts_to_tables(artifacts: GraphArtifacts) -> list[list[Row]]:
    embeddings = {record.owner_id: record.vector for record in artifacts.embeddings}
    doc_rows = [{"node_id": d.document_id, "type": "Document", "text": d.text, "title": d.title} for d in artifacts.documents]
    chunk_rows = [{"node_id": c.chunk_id, "type": "Chunk", "text": c.text, "document_id": c.document_id, "chunk_index": c.chunk_index, "embedding": embeddings.get(c.chunk_id), "pagerank": 0.0, "community": -1} for c in artifacts.chunks]
    entity_rows = [{"node_id": e.entity_id, "type": "Entity", "name": e.name, "entity_type": e.entity_type, "embedding": embeddings.get(e.entity_id), "pagerank": 0.0, "community": -1} for e in artifacts.entities]
    emb_rows = [{"node_id": f"emb:{r.owner_id}", "owner_id": r.owner_id, "vector": r.vector, "type": "Embedding"} for r in artifacts.embeddings]
    return [doc_rows, chunk_rows, entity_rows, emb_rows]

def edge_rows(artifacts: GraphArtifacts) -> list[Row]:
    rows = []
    rows.extend([{"src": c.document_id, "dst": c.chunk_id, "type": "HAS_CHUNK", "weight": 1.0} for c in artifacts.chunks])
    rows.extend([{"src": m.chunk_id, "dst": m.entity_id, "type": "MENTIONS", "weight": m.confidence} for m in artifacts.mentions])
    rows.extend([{"src": r.source_entity_id, "dst": r.target_entity_id, "type": r.relationship_type, "weight": r.weight} for r in artifacts.relationships])
    return rows

def result_to_rows(result: Any) -> list[Row]:
    if result is None: return []
    if hasattr(result, "arrow") and callable(result.arrow): return result.arrow().to_pylist()
    if hasattr(result, "to_pylist") and callable(result.to_pylist): return result.to_pylist()
    if isinstance(result, list): return result
    try: return list(result)
    except Exception: return []

def rows_to_arrow(rows: list[Row]) -> pa.Table:
    if not rows: return pa.table({})
    keys = sorted({k for r in rows for k in r})
    data = [{k: r.get(k) for k in keys} for r in rows]
    return pa.Table.from_pylist(data)

def append_note(existing: str, note: str) -> str:
    return note if not existing else f"{existing}; {note}"
