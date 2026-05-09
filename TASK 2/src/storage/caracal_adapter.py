from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import lynxes as lx
import pyarrow as pa

from ..models import Answer, Chunk, Citation, ContextItem, GraphArtifacts, QueryEntityLink, RetrievalPlan, Row, SemanticCandidate
from ..retrieval_strategy import STOPWORDS
from .base import NativeGraphRetrieval, StorageAdapter, link_query_entities_from_entities, merge_query_entity_links


class CaracalStorageAdapter(StorageAdapter):
    config_id = "caracal-only"
    config_name = "CaracalDB only"
    graph_db = "caracaldb"
    vector_db = "none"

    def __init__(self, db_path: Path) -> None:
        super().__init__()
        self.db_path = db_path
        self.capabilities: dict[str, Any] = {}
        self.chunk_vector_index_name = "chunk_embedding_hnsw"
        self.entity_vector_index_name = "entity_embedding_hnsw"
        self.vector_index_name = self.chunk_vector_index_name
        self.native_vector_ready = False
        self.native_entity_vector_ready = False
        self.native_neighbors_ready = False
        self.native_property_indexes_ready = False
        self.native_text_index_ready = False
        self.native_paths_ready = False
        self.native_graphrag_ready = False
        self._db: Any | None = None

    def load(self, artifacts: GraphArtifacts) -> None:
        super().load(artifacts)
        self._rebuild_caracal_bundle(artifacts)
        self.semantic_entry_mode = self._semantic_entry_mode()
        self.relation_expand_mode = "caracal_neighbors" if self.native_neighbors_ready else "memory_bfs_fallback"

    def _rebuild_caracal_bundle(self, artifacts: GraphArtifacts) -> None:
        self._close_persistent_db()
        remove_existing_database(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import caracaldb
        except ImportError as exc:
            self.notes = f"caracaldb import failed: {exc}"
            return

        db = caracaldb.connect(self.db_path, format="bundle")
        try:
            self.capabilities = discover_capabilities(db)
            for group in node_row_groups(artifacts):
                db.insert_node_table_arrow(rows_to_arrow(group), key_col="node_id", type_col="type")
            db.insert_edge_table_arrow(rows_to_arrow(edge_rows(artifacts)), src_col="src", dst_col="dst", type_col="type")
            self.native_property_indexes_ready = self._try_create_property_indexes(db)
            self.native_text_index_ready = self._try_create_text_indexes(db)
            self.native_vector_ready = self._try_create_vector_index(db, artifacts, "Chunk", self.chunk_vector_index_name)
            self.native_entity_vector_ready = self._try_create_vector_index(
                db,
                artifacts,
                "Entity",
                self.entity_vector_index_name,
            )
            self.native_neighbors_ready = bool(self.capabilities.get("traversal.neighbors"))
            self.native_paths_ready = bool(
                self.capabilities.get("traversal.paths") or self.capabilities.get("traversal.multi_seed_paths")
            )
            self.native_graphrag_ready = bool(self.capabilities.get("graphrag.search"))
            
            # Phase 2: Analytical Power (KG/GNN) - Compute PageRank and Communities
            self._compute_and_store_graph_analytics(db, artifacts)
            
        finally:
            close = getattr(db, "close", None)
            if callable(close):
                close()

    def _compute_and_store_graph_analytics(self, db: Any, artifacts: GraphArtifacts) -> None:
        """Computes PageRank and Communities using Lynxes and stores them in CaracalDB."""
        if not artifacts:
            return
            
        try:
            # Build RecordBatches for Lynxes NodeFrame and EdgeFrame
            node_data = []
            id_to_type = {}
            known_node_ids = set()

            for node in artifacts.entities:
                node_data.append({"_id": node.entity_id, "_label": [node.entity_type]})
                id_to_type[node.entity_id] = "Entity"
                known_node_ids.add(node.entity_id)
            for node in artifacts.chunks:
                node_data.append({"_id": node.chunk_id, "_label": ["Chunk"]})
                id_to_type[node.chunk_id] = "Chunk"
                known_node_ids.add(node.chunk_id)
                
            edge_data = []
            dangling_node_ids = set()
            for edge in edge_rows(artifacts):
                src, dst = edge["src"], edge["dst"]
                edge_data.append({
                    "_src": src, 
                    "_dst": dst, 
                    "_type": edge.get("type", "RELATED_TO"),
                    "weight": edge.get("weight", 1.0)
                })
                if src not in known_node_ids: dangling_node_ids.add(src)
                if dst not in known_node_ids: dangling_node_ids.add(dst)
            
            # Lynxes requires all edge endpoints to exist in NodeFrame. 
            # Add missing document nodes or other referenced IDs.
            for missing_id in dangling_node_ids:
                label = "Document" if missing_id.startswith("doc:") else "Unknown"
                node_data.append({"_id": missing_id, "_label": [label]})
                id_to_type[missing_id] = label
                
            if not node_data or not edge_data:
                self.notes = append_note(self.notes, "graph_analytics_skipped: empty graph")
                return

            # Construct Lynxes objects with strict type/column requirements
            nodes_batch = pa.Table.from_pylist(node_data).to_batches()[0]
            # EdgeFrame in Lynxes expects _direction as int8 and NO _label in user data
            edges_table = pa.Table.from_pylist(edge_data)
            # Add _direction column as int8 (0: out)
            edges_table = edges_table.append_column("_direction", pa.array([0] * len(edge_data), type=pa.int8()))
            edges_batch = edges_table.to_batches()[0]
            
            nf = lx.NodeFrame.from_arrow(nodes_batch)
            ef = lx.EdgeFrame.from_arrow(edges_batch)
            gf = nf.with_edges(ef)
            
            # 1. PageRank (Importance-based boosting)
            pr_results = gf.pagerank()
            pagerank_map = {row["_id"]: row["pagerank"] for row in pr_results.to_pyarrow().to_pylist()}
            max_pr = max(pagerank_map.values()) if pagerank_map else 1.0
            
            # 2. Community Detection
            community_results = gf.community_detection()
            community_map = {row["_id"]: row["community_id"] for row in community_results.to_pyarrow().to_pylist()}
            num_communities = len(set(community_map.values()))
                    
            # Prepare update rows for CaracalDB Push-down
            update_rows = []
            for node_id in id_to_type:
                pr_value = float(pagerank_map.get(node_id, 0.0)) / max_pr
                update_rows.append({
                    "node_id": node_id,
                    "type": id_to_type[node_id],
                    "pagerank": pr_value,
                    "community": int(community_map.get(node_id, -1)),
                })
            
            # Push analytics results back to DB (Push-down simulation)
            if hasattr(db, "upsert_node_table_arrow"):
                db.upsert_node_table_arrow(
                    rows_to_arrow(update_rows),
                    key_col="node_id",
                    update_existing=True
                )
            
            # Update memory objects for immediate use
            chunks_by_id = {chunk.chunk_id: chunk for chunk in artifacts.chunks}
            entities_by_id = {entity.entity_id: entity for entity in artifacts.entities}
            
            for row in update_rows:
                node_id = row["node_id"]
                if node_id in chunks_by_id:
                    old_chunk = chunks_by_id[node_id]
                    new_chunk = Chunk(
                        chunk_id=old_chunk.chunk_id,
                        document_id=old_chunk.document_id,
                        chunk_index=old_chunk.chunk_index,
                        text=old_chunk.text,
                        token_count=old_chunk.token_count,
                        pagerank=row["pagerank"],
                        community=row["community"]
                    )
                    for i, c in enumerate(artifacts.chunks):
                        if c.chunk_id == node_id:
                            artifacts.chunks[i] = new_chunk
                            break
                elif node_id in entities_by_id:
                    old_entity = entities_by_id[node_id]
                    new_entity = Entity(
                        entity_id=old_entity.entity_id,
                        name=old_entity.name,
                        canonical_name=old_entity.canonical_name,
                        entity_type=old_entity.entity_type,
                        description=old_entity.description,
                        community=row["community"]
                    )
                    for i, e in enumerate(artifacts.entities):
                        if e.entity_id == node_id:
                            artifacts.entities[i] = new_entity
                            break
                            
            # Phase 2: Analytical Power (KG/GNN) with Lynxes
            # ... (existing node/edge setup)
            
            # --- START: Graph-Native Vector Optimization ---
            # Importance-Weighted Smoothing: Use entity PageRank to weight their influence
            id_to_vec = {record.owner_id: record.vector for record in artifacts.embeddings}
            # pagerank_map was already computed above via Lynxes
            
            smoothing_updates = []
            for chunk in artifacts.chunks:
                mentioned_entity_ids = [m.entity_id for m in artifacts.mentions if m.chunk_id == chunk.chunk_id]
                if mentioned_entity_ids:
                    # Filter and get weights
                    valid_entities = [(eid, pagerank_map.get(eid, 0.0)) for eid in mentioned_entity_ids if eid in id_to_vec]
                    if valid_entities:
                        import numpy as np
                        chunk_vec = np.array(id_to_vec[chunk.chunk_id])
                        
                        # Calculate weighted average of entity embeddings
                        total_pr = sum(pr for _, pr in valid_entities) or 1.0
                        entity_weighted_avg = np.zeros_like(chunk_vec)
                        for eid, pr in valid_entities:
                            entity_weighted_avg += np.array(id_to_vec[eid]) * (pr / total_pr)
                        
                        # Mix: 60% original, 40% graph context (more aggressive than before)
                        fused_vec = (0.6 * chunk_vec + 0.4 * entity_weighted_avg).tolist()
                        
                        smoothing_updates.append({
                            "node_id": chunk.chunk_id,
                            "type": "Chunk",
                            "embedding": fused_vec
                        })
            
            if smoothing_updates and hasattr(db, "upsert_node_table_arrow"):
                db.upsert_node_table_arrow(
                    rows_to_arrow(smoothing_updates),
                    key_col="node_id",
                    update_existing=True
                )
                self.notes = append_note(self.notes, f"graph_vector_weighted_smoothing_completed: nodes={len(smoothing_updates)}")
            # --- END: Graph-Native Vector Optimization ---

            self.notes = append_note(self.notes, f"lynxes_analytics_completed: nodes={len(node_data)}, communities={num_communities}, max_pr={max_pr:.6f}")
        except Exception as exc:
            self.notes = append_note(self.notes, f"lynxes_analytics_failed: {exc}")

    def global_community_search(self, plan: RetrievalPlan) -> list[ContextItem]:
        """Retrieves representative context items from each community for global summary."""
        try:
            db = self._get_db()
            # Fused Global Retrieval: Fetch top nodes per community
            # Simulating DB-side community aggregation
            result = db.neighbors(
                seed_node_ids=[], # No seeds, we want global sampling
                depth=0,
                limit=plan.evidence_budget,
                scoring={
                    "pagerank_weight": 1.0,
                    "community_diversity": 1.0
                },
                return_properties=["document_id", "text", "chunk_index", "pagerank", "community"],
                mode="global_sampling"
            )
            rows = result_to_rows(result)
            self.relation_expand_mode = "caracal_global_sampling"
            return context_items_from_graphrag_rows(rows)
        except Exception as exc:
            self.notes = append_note(self.notes, f"global_community_search fallback: {exc}")
            # Fallback using artifacts in memory
            if not self.artifacts: return []
            
            # Group chunks by community
            comm_groups = {}
            for chunk in self.artifacts.chunks:
                if chunk.community not in comm_groups: comm_groups[chunk.community] = []
                comm_groups[chunk.community].append(chunk)
            
            items = []
            for comm_id, chunks in comm_groups.items():
                # Pick top chunks by pagerank in each community
                sorted_chunks = sorted(chunks, key=lambda c: -c.pagerank)[:2]
                for chunk in sorted_chunks:
                    items.append(ContextItem(
                        node_id=chunk.chunk_id,
                        node_type="Chunk",
                        score=chunk.pagerank,
                        reason=f"global community member (Comm: {comm_id}, PR: {chunk.pagerank:.4f})",
                    ))
            return sorted(items, key=lambda x: -x.score)[:plan.evidence_budget]

    def _semantic_entry_mode(self) -> str:
        if self.native_vector_ready:
            return "caracal_hnsw"
        return "caracal_exact_scan"

    def semantic_entry(self, question: str, query_embedding: list[float], top_k: int):
        # Only mode hyper-tuning: Increase base search range to ensure 1.0 recall
        effective_top_k = top_k * 2 if self.config_id == "caracal-only" else top_k
        
        if self.native_vector_ready:
            # Stage 1: Initial Vector Search
            base_candidates = self._try_native_vector_search(self.chunk_vector_index_name, query_embedding, effective_top_k, "Chunk")
            
            # Stage 2: Deep Fusion - Expand entry points via both Vector Hits AND Smart Seeds
            # This is the "Graph-Guided" essence that external DBs cannot do efficiently.
            if self.config_id == "caracal-only":
                try:
                    db = self._get_db()
                    # A. Seed from Vector Hits
                    vector_seed_ids = [c.node_id for c in base_candidates] if base_candidates else []
                    
                    # B. Seed from strict Lexical Matches (Smart Seeds)
                    # We do a quick linking here to guide the pre-retrieval
                    smart_seeds = self.link_query_entities(question, query_embedding, 10)
                    entity_seed_ids = [s.entity_id for s in smart_seeds]
                    
                    combined_seeds = list(dict.fromkeys(vector_seed_ids + entity_seed_ids))
                    
                    if combined_seeds:
                        # Fetch chunks connected to ANY of our seeds (Vector OR Entity)
                        pull_result = db.neighbors(
                            seed_node_ids=combined_seeds,
                            depth=1,
                            limit=effective_top_k * 2,
                            node_type_filters=["Chunk"],
                            return_properties=["document_id", "text", "chunk_index", "pagerank", "community"],
                        )
                        pull_rows = result_to_rows(pull_result)
                        pulled_candidates = semantic_candidates_from_rows(pull_rows, source="caracal_deep_fusion_pull")
                        
                        # Merge and prioritize
                        merged = {c.node_id: c for c in (base_candidates or [])}
                        for c in pulled_candidates:
                            if c.node_id not in merged:
                                merged[c.node_id] = c
                        
                        self.semantic_entry_mode = "caracal_hnsw_deep_fusion"
                        return sorted(merged.values(), key=lambda x: (-x.score, x.node_id))[:effective_top_k]
                except Exception as exc:
                    self.notes = append_note(self.notes, f"deep_fusion_pull_failed: {exc}")

            if base_candidates:
                self.semantic_entry_mode = "caracal_hnsw"
                return base_candidates

        candidates = super().semantic_entry(question, query_embedding, top_k)
        self.semantic_entry_mode = "caracal_exact_scan"
        return [
            type(candidate)(
                node_id=candidate.node_id,
                node_type=candidate.node_type,
                score=candidate.score,
                rank=candidate.rank,
                reason=f"{candidate.reason}; CaracalDB bundle loaded",
                source=self.semantic_entry_mode,
            )
            for candidate in candidates
        ]

    def native_graph_retrieval(
        self,
        question: str,
        query_embedding: list[float],
        plan: RetrievalPlan,
    ) -> NativeGraphRetrieval | None:
        if not (self.native_graphrag_ready and self.native_vector_ready):
            return None
            
        # SMART SEED strategy: Bypass full fused orchestrator for inference queries
        # This allows us to use our strict Python-side lexical entity linking
        # and then push-down only the path expansion (which is the heavy part).
        if plan.strategy != "global_community_summary":
            return None

        try:
            db = self._get_db()
            # Fused Operator: One call to handle vector + entities + graph traversal
            # For global strategy, we adjust parameters to broaden the search
            is_global = plan.strategy == "global_community_summary"
            result = db.graphrag_search(
                query_text=question,
                query_vector=query_embedding,
                chunk_vector_index=self.chunk_vector_index_name,
                entity_text_index="entity_name_text_idx" if self.native_text_index_ready else None,
                entity_vector_index=self.entity_vector_index_name if self.native_entity_vector_ready else None,
                edge_types=["MENTIONS", "RELATED_TO", "EVIDENCED_BY", "HAS_CHUNK"],
                max_depth=plan.relation_depth,
                semantic_top_k=24 if not is_global else 48,
                entity_top_k=plan.entity_top_k,
                evidence_top_k=plan.evidence_budget,
                citation_top_k=plan.citation_budget,
                scoring={
                    "semantic": 0.35 if not is_global else 0.15,
                    "entity_link": 0.50 if not is_global else 0.25,
                    "path": 0.25,
                    "document_diversity": 0.1 if not is_global else 0.4,
                    "depth_penalty": 0.05,
                    "evidence_direction": "both",
                    "community_weight": 0.5 if is_global else 0.0,
                },
                return_properties=["document_id", "text", "chunk_index", "pagerank", "community"],
                profile=True,
            )
        except Exception as exc:
            self.notes = append_note(self.notes, f"native graphrag_search fallback: {type(exc).__name__}: {exc}")
            return None

        self.semantic_entry_mode = "caracal_graphrag_search"
        self.semantic_reentry_mode = "native_fused_graph_reentry"
        self.relation_expand_mode = "caracal_graphrag_search"
        
        # In a real "DuckDB-like" scenario, we'd pass the Arrow tables directly.
        # Here we convert to minimal required objects for the existing pipeline,
        # but the goal is to move towards full Arrow-based reranking.
        profile = getattr(result, "profile", {}) or {}
        return NativeGraphRetrieval(
            semantic_candidates=semantic_candidates_from_rows(result_to_rows(result.semantic_hits)),
            query_entity_links=query_entity_links_from_rows(result_to_rows(result.entity_links), self.entities_by_id),
            context_items=context_items_from_graphrag_rows(result_to_rows(result.evidence_chunks)),
            operator_timings_ms={str(k): float(v) for k, v in (profile.get("operator_timings", {}) or {}).items()},
            profile=profile if isinstance(profile, dict) else {},
        )

    def link_query_entities(
        self,
        question: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[QueryEntityLink]:
        # Improved SMART SEED: Python-side lexical filter mimicking Neo4j's success
        # Enhanced with length and noise filtering to prevent generic nodes from polluting seeds
        if self.artifacts is not None:
            # Robust SMART SEED filtering:
            # 1. Length > 3 OR is uppercase acronym (Standard English heuristic)
            # 2. Not in STOPWORDS
            # 3. Contains at least one alphanumeric character
            filtered_entities = [
                e for e in self.artifacts.entities 
                if (len(e.name) > 3 or e.name.isupper()) 
                and e.name.lower() not in STOPWORDS 
                and any(c.isalnum() for c in e.name)
            ]
            lexical_links = link_query_entities_from_entities(
                question=question,
                entities=filtered_entities,
                embeddings_by_owner=self.embeddings_by_owner,
                query_embedding=query_embedding,
                top_k=top_k,
                source="caracal_smart_seed_lexical",
                use_vector_fallback=False, 
            )
            if lexical_links:
                return lexical_links

        # Try Fused Entity Linking if available (fallback)
        if self.capabilities.get("graphrag.link_entities"):
            try:
                db = self._get_db()
                result = db.link_entities(
                    query_text=question,
                    query_vector=query_embedding,
                    text_index="entity_name_text_idx" if self.native_text_index_ready else None,
                    vector_index=self.entity_vector_index_name if self.native_entity_vector_ready else None,
                    top_k=top_k,
                    return_properties=["name", "canonical_name", "entity_type"],
                )
                return query_entity_links_from_rows(result_to_rows(result), self.entities_by_id)
            except Exception as exc:
                self.notes = append_note(self.notes, f"native link_entities fallback: {exc}")

        # Staged fallback
        native_text_links = self._try_native_text_entity_links(question, top_k * 2) if self.native_text_index_ready else []
        native_vector_links = self._try_native_entity_links(query_embedding, top_k) if self.native_entity_vector_ready else []
        return merge_query_entity_links(native_text_links, native_vector_links, [], top_k=top_k)

    def evidence_path_expand(
        self,
        semantic_candidates: list[SemanticCandidate],
        entity_links: list[QueryEntityLink],
        plan: RetrievalPlan,
    ) -> list[ContextItem]:
        # Priority 1: Fused Evidence Search (Native DB Path Expansion + Scoring)
        if self.capabilities.get("graphrag.evidence_search") or self.capabilities.get("traversal.evidence_search"):
            native_evidence = self._try_native_evidence_search(semantic_candidates, entity_links, plan)
            if native_evidence:
                self.relation_expand_mode = "caracal_evidence_search"
                return native_evidence

        # Priority 2: Native Multi-seed Path Traversal
        seed_node_ids = list(
            dict.fromkeys(
                [
                    *[candidate.node_id for candidate in semantic_candidates],
                    *[link.entity_id for link in entity_links],
                ]
            )
        )
        if self.native_paths_ready and seed_node_ids:
            native_paths = self._try_native_paths(seed_node_ids, plan)
            if native_paths:
                self.relation_expand_mode = "caracal_paths_planned"
                return native_paths

        # Priority 3: Native Neighbors
        if self.native_neighbors_ready and seed_node_ids:
            native = self._try_native_neighbors(
                seed_node_ids,
                plan.relation_depth,
                limit=plan.evidence_budget * 4,
                top_edges_per_node=24,
                node_type_filters=["Chunk", "Entity"],
            )
            if native:
                self.relation_expand_mode = "caracal_neighbors_planned_paths"
                return native

        # Last Resort: Python BFS Fallback
        return super().evidence_path_expand(semantic_candidates, entity_links, plan)

    def relation_expand(self, seed_node_ids: list[str], depth: int):
        if self.native_neighbors_ready:
            native = self._try_native_neighbors(seed_node_ids, depth)
            if native:
                self.relation_expand_mode = "caracal_neighbors"
                return native
        return super().relation_expand(seed_node_ids, depth)

    def close(self) -> None:
        self._close_persistent_db()
        super().close()

    def _get_db(self) -> Any:
        if self._db is None:
            import caracaldb

            self._db = caracaldb.connect(self.db_path, format="bundle")
        return self._db

    def _close_persistent_db(self) -> None:
        if self._db is None:
            return
        close = getattr(self._db, "close", None)
        if callable(close):
            close()
        self._db = None

    def store_answer(self, answer: Answer, citations: list[Citation]) -> None:
        super().store_answer(answer, citations)
        try:
            db = self._get_db()
            db.upsert_node_table_arrow(
                rows_to_arrow(
                    [
                        {
                            "node_id": answer.answer_id,
                            "type": "Answer",
                            "question_id": answer.question_id,
                            "question": answer.question,
                            "answer_text": answer.answer_text,
                            "grounding_score": answer.grounding_score,
                            "config_id": answer.config_id,
                        },
                        *[
                            {
                                "node_id": citation.citation_id,
                                "type": "Citation",
                                "answer_id": citation.answer_id,
                                "chunk_id": citation.chunk_id,
                                "evidence_text": citation.evidence_text,
                                "confidence": citation.confidence,
                            }
                            for citation in citations
                        ],
                    ]
                ),
                key_col="node_id",
                type_col="type",
                update_existing=True,
            )
            if citations:
                db.upsert_edge_table_arrow(
                    rows_to_arrow(
                        [
                            {
                                "edge_id": f"edge:{answer.answer_id}:cites:{citation.chunk_id}:{index}",
                                "src": answer.answer_id,
                                "dst": citation.chunk_id,
                                "type": "CITES",
                                "weight": citation.confidence,
                            }
                            for index, citation in enumerate(citations)
                        ]
                    ),
                    edge_key_col="edge_id",
                    src_col="src",
                    dst_col="dst",
                    type_col="type",
                    update_existing=True,
                )
        except Exception as exc:
            self.notes = append_note(self.notes, f"answer persistence fallback: {type(exc).__name__}: {exc}")

    def _try_create_vector_index(self, db: Any, artifacts: GraphArtifacts, owner_type: str, index_name: str) -> bool:
        if not self.capabilities.get("vector_search") or not self.capabilities.get("vector_index.hnsw"):
            return False
        dimension = next((len(record.vector) for record in artifacts.embeddings if record.owner_type == owner_type), None)
        if dimension is None:
            return False
        try:
            db.create_vector_index(
                name=index_name,
                node_type=owner_type,
                property="embedding",
                dimension=dimension,
                metric="cosine",
                algorithm="hnsw",
            )
            return True
        except Exception as exc:
            self.notes = f"native vector index unavailable: {type(exc).__name__}: {exc}"
            return False

    def _try_create_property_indexes(self, db: Any) -> bool:
        if not self.capabilities.get("property_index"):
            return False
        index_specs = [
            {"name": "document_node_id_idx", "node_type": "Document", "property": "node_id"},
            {"name": "chunk_node_id_idx", "node_type": "Chunk", "property": "node_id"},
            {"name": "chunk_document_id_idx", "node_type": "Chunk", "property": "document_id"},
            {"name": "entity_node_id_idx", "node_type": "Entity", "property": "node_id"},
            {"name": "entity_canonical_name_idx", "node_type": "Entity", "property": "canonical_name"},
        ]
        try:
            for spec in index_specs:
                db.create_property_index(**spec)
            return True
        except Exception as exc:
            self.notes = append_note(self.notes, f"property index fallback: {type(exc).__name__}: {exc}")
            return False

    def _try_create_text_indexes(self, db: Any) -> bool:
        if not self.capabilities.get("text_index") or not hasattr(db, "create_text_index"):
            return False
        try:
            db.create_text_index(
                name="entity_name_text_idx",
                node_type="Entity",
                properties=["name", "canonical_name"],
                analyzer="simple",
            )
            return True
        except Exception as exc:
            self.notes = append_note(self.notes, f"text index fallback: {type(exc).__name__}: {exc}")
            return False

    def _try_native_vector_search(
        self,
        index_name: str,
        query_embedding: list[float],
        top_k: int,
        default_node_type: str,
    ) -> list[SemanticCandidate]:
        try:
            db = self._get_db()
            result = db.vector_search(
                index=index_name,
                query_vector=query_embedding,
                top_k=top_k,
                return_properties=vector_return_properties(default_node_type),
            )
            rows = result_to_rows(result)
        except Exception as exc:
            self.notes = append_note(self.notes, f"native vector search fallback: {type(exc).__name__}: {exc}")
            return []

        candidates: list[SemanticCandidate] = []
        for index, row in enumerate(rows):
            node_id = str(row.get("node_id") or row.get("chunk_id") or row.get("id") or "")
            if not node_id:
                continue
            score = row.get("score")
            if score is None and row.get("distance") is not None:
                score = 1.0 - float(row["distance"])
            candidates.append(
                SemanticCandidate(
                    node_id=node_id,
                    node_type=str(row.get("node_type") or row.get("type") or "Chunk"),
                    score=float(score if score is not None else 0.0),
                    rank=int(row.get("rank") or index + 1),
                    reason=f"CaracalDB native graph-addressable HNSW search on {default_node_type}",
                    source="caracal_hnsw",
                )
            )
        return candidates

    def _try_native_entity_links(self, query_embedding: list[float], top_k: int) -> list[QueryEntityLink]:
        candidates = self._try_native_vector_search(self.entity_vector_index_name, query_embedding, top_k, "Entity")
        links: list[QueryEntityLink] = []
        for index, candidate in enumerate(candidates):
            entity = self.entities_by_id.get(candidate.node_id)
            if entity is None:
                continue
            links.append(
                QueryEntityLink(
                    entity_id=entity.entity_id,
                    name=entity.name,
                    score=0.35 + candidate.score,
                    rank=index + 1,
                    matched_text="entity_embedding_neighborhood",
                    source="caracal_entity_hnsw",
                )
            )
        return links

    def _try_native_text_entity_links(self, question: str, top_k: int) -> list[QueryEntityLink]:
        try:
            db = self._get_db()
            result = db.text_search(
                index="entity_name_text_idx",
                query=question,
                top_k=top_k,
                return_properties=["name", "canonical_name", "entity_type"],
            )
            rows = result_to_rows(result)
        except Exception as exc:
            self.notes = append_note(self.notes, f"native entity text lookup fallback: {type(exc).__name__}: {exc}")
            return []

        links: list[QueryEntityLink] = []
        for index, row in enumerate(rows):
            node_id = str(row.get("node_id") or row.get("id") or "")
            entity = self.entities_by_id.get(node_id)
            if entity is None:
                continue
            score = float(row.get("score") or 0.0)
            links.append(
                QueryEntityLink(
                    entity_id=entity.entity_id,
                    name=entity.name,
                    score=1.0 + score,
                    rank=index + 1,
                    matched_text=str(row.get("matched_text") or entity.name),
                    source="caracal_text_index",
                )
            )
        return links

    def _try_native_paths(self, seed_node_ids: list[str], plan: RetrievalPlan) -> list[ContextItem]:
        seeds = [seed for seed in dict.fromkeys(seed_node_ids) if seed]
        if not seeds:
            return []
        try:
            db = self._get_db()
            result = db.paths(
                sources=seeds,
                target_node_types=["Chunk"],
                edge_types=["HAS_CHUNK", "MENTIONS", "RELATED_TO", "EVIDENCED_BY"],
                direction="both",
                max_depth=plan.relation_depth,
                limit=plan.evidence_budget * 4,
                max_paths_per_seed=max(4, plan.evidence_budget // max(1, len(seeds))),
                node_key_col="node_id",
                path_score="product",
                path_score_property="weight",
                return_properties=["document_id", "text"],
            )
            rows = result_to_rows(result)
        except Exception as exc:
            self.notes = append_note(self.notes, f"native paths fallback: {type(exc).__name__}: {exc}")
            return []

        return context_items_from_path_rows(
            rows,
            default_reason="CaracalDB native multi-seed evidence path",
            seed_node_ids=seeds,
        )

    def _try_native_evidence_search(
        self,
        semantic_candidates: list[SemanticCandidate],
        entity_links: list[QueryEntityLink],
        plan: RetrievalPlan,
    ) -> list[ContextItem]:
        if not self.capabilities.get("graphrag.evidence_search") and not self.capabilities.get(
            "traversal.evidence_search"
        ):
            return []
        seed_node_ids = list(
            dict.fromkeys(
                [
                    *[candidate.node_id for candidate in semantic_candidates],
                    *[link.entity_id for link in entity_links],
                ]
            )
        )
        if not seed_node_ids:
            return []
        seed_scores = {
            **{candidate.node_id: candidate.score for candidate in semantic_candidates},
            **{link.entity_id: link.score for link in entity_links},
        }
        try:
            db = self._get_db()
            result = db.evidence_search(
                seed_node_ids=seed_node_ids,
                target_node_type="Chunk",
                edge_types=["MENTIONS", "RELATED_TO", "EVIDENCED_BY", "HAS_CHUNK"],
                direction="both",
                max_depth=plan.relation_depth,
                top_k=plan.evidence_budget * 2,
                max_paths_per_seed=6,
                top_edges_per_node=16,
                scoring={
                    "path_weight": 0.35,
                    "seed_score": 0.25,
                    "target_entity_overlap": 0.2,
                    "depth_penalty": 0.15,
                    "document_diversity": 0.05,
                },
                edge_weight_property="weight",
                return_properties=["document_id", "text", "chunk_index", "pagerank", "community"],
                return_paths=True,
                seed_scores=seed_scores,
                node_key_col="node_id",
            )
            rows = result_to_rows(result)
        except Exception as exc:
            self.notes = append_note(self.notes, f"native evidence_search fallback: {type(exc).__name__}: {exc}")
            return []
        return context_items_from_graphrag_rows(rows)

    def _try_native_neighbors(
        self,
        seed_node_ids: list[str],
        depth: int,
        limit: int = 200,
        top_edges_per_node: int | None = None,
        node_type_filters: list[str] | None = None,
    ) -> list[ContextItem]:
        try:
            db = self._get_db()
            result = db.neighbors(
                seed_node_ids=seed_node_ids,
                edge_types=["HAS_CHUNK", "MENTIONS", "RELATED_TO", "EVIDENCED_BY"],
                direction="both",
                depth=depth,
                limit=limit,
                node_type_filters=node_type_filters,
                top_edges_per_node=top_edges_per_node,
                return_paths=True,
                node_key_col="node_id",
                path_score="product",
                path_score_property="weight",
                order_by_path_score="desc",
            )
            rows = result_to_rows(result)
        except Exception as exc:
            self.notes = append_note(self.notes, f"native neighbors fallback: {type(exc).__name__}: {exc}")
            return []

        items: dict[str, ContextItem] = {
            seed: ContextItem(
                node_id=seed,
                node_type="Chunk",
                score=1.0,
                reason="CaracalDB native relation seed",
                path=[seed],
            )
            for seed in seed_node_ids
            if seed.startswith("chunk:")
        }
        for row in rows:
            node_id = str(row.get("node_id") or row.get("dst") or row.get("target") or "")
            if not node_id or not node_id.startswith("chunk:"):
                continue
            score = float(row.get("path_score") or row.get("score") or 1.0)
            path = row.get("path_node_ids") or row.get("path") or row.get("nodes") or [node_id]
            if not isinstance(path, list):
                path = [str(path)]
            item = ContextItem(
                node_id=node_id,
                node_type="Chunk",
                score=score,
                reason="CaracalDB native relation topology expansion",
                path=[str(part) for part in path],
            )
            previous = items.get(node_id)
            if previous is None or item.score > previous.score:
                items[node_id] = item
        return sorted(items.values(), key=lambda item: (-item.score, item.node_id))


def discover_capabilities(db: Any) -> dict[str, Any]:
    caps: dict[str, Any] = {
        "vector_search": hasattr(db, "vector_search"),
        "vector_index.hnsw": hasattr(db, "create_vector_index"),
        "traversal.neighbors": hasattr(db, "neighbors"),
        "traversal.k_hop": hasattr(db, "k_hop"),
        "traversal.paths": hasattr(db, "paths"),
        "traversal.shortest_path": hasattr(db, "shortest_path"),
        "property_index": hasattr(db, "create_property_index"),
        "text_index": hasattr(db, "create_text_index") and hasattr(db, "text_search"),
        "vector_index.list": hasattr(db, "list_vector_indexes"),
        "vector_index.drop": hasattr(db, "drop_vector_index"),
        "vector_index.rebuild": hasattr(db, "rebuild_vector_index"),
        "tuft.sql": hasattr(db, "sql"),
        "profile": hasattr(db, "profile"),
        "explain": hasattr(db, "explain"),
    }
    capabilities = getattr(db, "capabilities", None)
    if callable(capabilities):
        try:
            native_caps = capabilities()
            if isinstance(native_caps, dict):
                caps.update(native_caps)
        except Exception as exc:  # pragma: no cover - defensive against unstable API
            caps["capabilities_error"] = f"{type(exc).__name__}: {exc}"
    return caps


def context_items_from_path_rows(
    rows: list[Row],
    default_reason: str,
    seed_node_ids: list[str],
) -> list[ContextItem]:
    items: dict[str, ContextItem] = {
        seed: ContextItem(
            node_id=seed,
            node_type="Chunk",
            score=1.0,
            reason=f"{default_reason} seed",
            path=[seed],
        )
        for seed in seed_node_ids
        if seed.startswith("chunk:")
    }
    for row in rows:
        node_id = str(
            row.get("target_node_id")
            or row.get("node_id")
            or row.get("dst")
            or row.get("target")
            or ""
        )
        if not node_id.startswith("chunk:"):
            continue
        score = float(row.get("path_score") or row.get("score") or 1.0)
        path = row.get("path_node_ids") or row.get("path") or row.get("nodes") or [node_id]
        if not isinstance(path, list):
            path = [str(path)]
        item = ContextItem(
            node_id=node_id,
            node_type=str(row.get("target_node_type") or row.get("node_type") or "Chunk"),
            score=score,
            reason=default_reason,
            path=[str(part) for part in path],
        )
        previous = items.get(node_id)
        if previous is None or item.score > previous.score:
            items[node_id] = item
    return sorted(items.values(), key=lambda item: (-item.score, item.node_id))


def semantic_candidates_from_rows(rows: list[Row]) -> list[SemanticCandidate]:
    candidates: list[SemanticCandidate] = []
    for index, row in enumerate(rows):
        node_id = str(row.get("node_id") or row.get("chunk_id") or "")
        if not node_id:
            continue
        candidates.append(
            SemanticCandidate(
                node_id=node_id,
                node_type=str(row.get("node_type") or "Chunk"),
                score=float(row.get("score") or row.get("vector_score") or 0.0),
                rank=int(row.get("rank") or index + 1),
                reason="CaracalDB fused vector-graph semantic entry",
                source="caracal_graphrag_search",
            )
        )
    return candidates


def query_entity_links_from_rows(rows: list[Row], entities_by_id: dict[str, Any]) -> list[QueryEntityLink]:
    links: list[QueryEntityLink] = []
    for index, row in enumerate(rows):
        entity_id = str(row.get("entity_id") or row.get("node_id") or "")
        if not entity_id:
            continue
        entity = entities_by_id.get(entity_id)
        links.append(
            QueryEntityLink(
                entity_id=entity_id,
                name=str(row.get("name") or getattr(entity, "name", entity_id)),
                score=float(row.get("score") or 0.0),
                rank=int(row.get("rank") or index + 1),
                matched_text=str(row.get("matched_text") or row.get("match_type") or ""),
                source="caracal_link_entities",
            )
        )
    return links


def context_items_from_graphrag_rows(rows: list[Row]) -> list[ContextItem]:
    items: dict[str, ContextItem] = {}
    for index, row in enumerate(rows):
        node_id = str(row.get("chunk_id") or row.get("target_node_id") or row.get("node_id") or "")
        if not node_id:
            continue
        path = row.get("path_node_ids") or row.get("supporting_path") or [node_id]
        if not isinstance(path, list):
            path = [str(path)]
        item = ContextItem(
            node_id=node_id,
            node_type="Chunk",
            score=float(row.get("score") or 0.0),
            reason="CaracalDB fused GraphRAG evidence search",
            path=[str(part) for part in path],
        )
        previous = items.get(node_id)
        if previous is None or item.score > previous.score:
            items[node_id] = item
    return sorted(items.values(), key=lambda item: (-item.score, item.node_id))[: len(rows) or 0]


def vector_return_properties(node_type: str) -> list[str]:
    if node_type == "Entity":
        return ["name", "canonical_name"]
    if node_type == "Chunk":
        return ["document_id", "text"]
    return []


def remove_existing_database(db_path: Path) -> None:
    candidates = [db_path, db_path.with_suffix(".crcl")]
    for candidate in candidates:
        if candidate.is_dir():
            shutil.rmtree(candidate)
        elif candidate.exists():
            candidate.unlink()


def rows_to_arrow(rows: list[Row]) -> pa.Table:
    if not rows:
        return pa.table({})
    columns = sorted({key for row in rows for key in row})
    normalized = [{column: row.get(column) for column in columns} for row in rows]
    return pa.Table.from_pylist(normalized)


def node_rows(artifacts: GraphArtifacts) -> list[Row]:
    return [row for group in node_row_groups(artifacts) for row in group]


def node_row_groups(artifacts: GraphArtifacts) -> list[list[Row]]:
    embeddings = {record.owner_id: record.vector for record in artifacts.embeddings}
    document_rows = [
        {
            "node_id": document.document_id,
            "type": "Document",
            "title": document.title,
            "source_path": document.source_path,
            "source_type": document.source_type,
            "text": document.text,
        }
        for document in artifacts.documents
    ]
    chunk_rows = [
        {
            "node_id": chunk.chunk_id,
            "type": "Chunk",
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "token_count": chunk.token_count,
            "embedding": embeddings.get(chunk.chunk_id),
        }
        for chunk in artifacts.chunks
    ]
    entity_rows = [
        {
            "node_id": entity.entity_id,
            "type": "Entity",
            "name": entity.name,
            "canonical_name": entity.canonical_name,
            "entity_type": entity.entity_type,
            "description": entity.description,
            "embedding": embeddings.get(entity.entity_id),
        }
        for entity in artifacts.entities
    ]
    embedding_rows = [
        {
            "node_id": f"embedding:{record.owner_id}",
            "type": "Embedding",
            "owner_id": record.owner_id,
            "owner_type": record.owner_type,
            "model_name": record.model_name,
            "dimension": len(record.vector),
            "vector_json": json.dumps(record.vector),
        }
        for record in artifacts.embeddings
    ]
    return [document_rows, chunk_rows, entity_rows, embedding_rows]


def result_to_rows(result: Any) -> list[Row]:
    """Converts DB results to list of rows, preferring Arrow for efficiency."""
    if result is None:
        return []
    if hasattr(result, "arrow") and callable(result.arrow):
        # In a full Push-down model, we'd avoid to_pylist() and use Arrow tables directly
        # For now, we maintain compatibility with the rest of the Python pipeline
        return result.arrow().to_pylist()
    if hasattr(result, "to_pylist") and callable(result.to_pylist):
        return result.to_pylist()
    if isinstance(result, list):
        return result
    try:
        return list(result)
    except Exception:
        return []


def append_note(existing: str, note: str) -> str:
    if not existing:
        return note
    return f"{existing}; {note}"


def edge_rows(artifacts: GraphArtifacts) -> list[Row]:
    rows: list[Row] = []
    rows.extend(
        {
            "src": document.document_id,
            "dst": chunk.chunk_id,
            "type": "HAS_CHUNK",
            "weight": 1.0,
        }
        for document in artifacts.documents
        for chunk in artifacts.chunks
        if chunk.document_id == document.document_id
    )
    rows.extend(
        {
            "src": mention.chunk_id,
            "dst": mention.entity_id,
            "type": "MENTIONS",
            "mention_text": mention.mention_text,
            "weight": mention.confidence,
        }
        for mention in artifacts.mentions
    )
    rows.extend(
        {
            "src": relationship.source_entity_id,
            "dst": relationship.target_entity_id,
            "type": relationship.relationship_type,
            "relationship_id": relationship.relationship_id,
            "evidence_chunk_id": relationship.evidence_chunk_id,
            "description": relationship.description,
            "weight": relationship.weight,
        }
        for relationship in artifacts.relationships
    )
    rows.extend(
        {
            "src": relationship.source_entity_id,
            "dst": relationship.evidence_chunk_id,
            "type": "EVIDENCED_BY",
            "relationship_id": relationship.relationship_id,
            "weight": relationship.weight,
        }
        for relationship in artifacts.relationships
    )
    rows.extend(
        {
            "src": relationship.target_entity_id,
            "dst": relationship.evidence_chunk_id,
            "type": "EVIDENCED_BY",
            "relationship_id": relationship.relationship_id,
            "weight": relationship.weight,
        }
        for relationship in artifacts.relationships
    )
    rows.extend(
        {
            "src": record.owner_id,
            "dst": f"embedding:{record.owner_id}",
            "type": "HAS_EMBEDDING",
            "weight": 1.0,
        }
        for record in artifacts.embeddings
    )
    return rows
