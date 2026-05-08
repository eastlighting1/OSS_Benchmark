from __future__ import annotations

import os

from ..models import ContextItem, GraphArtifacts, QueryEntityLink, RetrievalPlan, SemanticCandidate
from .base import MissingExternalService, StorageAdapter, merge_query_entity_links


class Neo4jStorageAdapter(StorageAdapter):
    config_id = "neo4j-only"
    config_name = "Neo4j only"
    graph_db = "neo4j"
    vector_db = "none"

    def __init__(self) -> None:
        super().__init__()
        self.driver = None
        self.vector_index_name = "chunk_embedding_index"
        self.entity_vector_index_name = "entity_embedding_index"

    def load(self, artifacts: GraphArtifacts) -> None:
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")
        if not uri:
            raise MissingExternalService("NEO4J_URI is not configured; Neo4j benchmark skipped")
        if not password:
            raise MissingExternalService("NEO4J_PASSWORD is not configured; Neo4j benchmark skipped")

        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise MissingExternalService(f"neo4j Python package is unavailable: {exc}") from exc

        super().load(artifacts)
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n").consume()
                self._create_lookup_indexes(session)
                self._load_nodes(session, artifacts)
                self._load_edges(session, artifacts)
                self._create_vector_indexes(session, artifacts)
        except Exception as exc:
            self.close()
            raise MissingExternalService(f"Neo4j connection/write failed: {type(exc).__name__}: {exc}") from exc
        self.semantic_entry_mode = "neo4j_vector_index"
        self.relation_expand_mode = "cypher_paths"

    def semantic_entry(self, question: str, query_embedding: list[float], top_k: int) -> list[SemanticCandidate]:
        if self.driver is None:
            return super().semantic_entry(question, query_embedding, top_k)
        with self.driver.session() as session:
            rows = session.run(
                """
                CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
                YIELD node, score
                RETURN node.node_id AS node_id, labels(node)[0] AS node_type, score
                ORDER BY score DESC, node.node_id ASC
                """,
                index_name=self.vector_index_name,
                top_k=top_k,
                embedding=query_embedding,
            ).data()
        self.semantic_entry_mode = "neo4j_vector_index"
        return [
            SemanticCandidate(
                node_id=row["node_id"],
                node_type=row["node_type"],
                score=float(row["score"]),
                rank=index + 1,
                reason="Neo4j native vector index semantic entry",
                source="neo4j_vector_index",
            )
            for index, row in enumerate(rows)
        ]

    def link_query_entities(
        self,
        question: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[QueryEntityLink]:
        lexical_links = super().link_query_entities(question, query_embedding, top_k)
        if self.driver is None:
            return lexical_links
        cypher_links: list[QueryEntityLink] = []
        with self.driver.session() as session:
            lexical_rows = session.run(
                """
                MATCH (e:Entity)
                WHERE toLower($question) CONTAINS toLower(e.name)
                   OR any(token IN split(toLower(e.name), ' ')
                          WHERE size(token) > 2 AND toLower($question) CONTAINS token)
                RETURN e.node_id AS entity_id, e.name AS name,
                       CASE WHEN toLower($question) CONTAINS toLower(e.name) THEN 1.4 ELSE 0.65 END AS score,
                       e.name AS matched_text
                ORDER BY score DESC, entity_id ASC
                LIMIT $limit
                """,
                question=question,
                limit=top_k * 2,
            ).data()
            vector_rows = session.run(
                """
                CALL db.index.vector.queryNodes($index_name, $limit, $embedding)
                YIELD node, score
                RETURN node.node_id AS entity_id, node.name AS name, score
                ORDER BY score DESC, entity_id ASC
                """,
                index_name=self.entity_vector_index_name,
                limit=top_k,
                embedding=query_embedding,
            ).data()
        for row in lexical_rows:
            cypher_links.append(
                QueryEntityLink(
                    entity_id=row["entity_id"],
                    name=row["name"],
                    score=float(row["score"]),
                    rank=0,
                    matched_text=row["matched_text"],
                    source="neo4j_entity_property_lookup",
                )
            )
        for row in vector_rows:
            cypher_links.append(
                QueryEntityLink(
                    entity_id=row["entity_id"],
                    name=row["name"],
                    score=0.35 + float(row["score"]),
                    rank=0,
                    matched_text="entity_vector_index",
                    source="neo4j_entity_vector_index",
                )
            )
        return merge_query_entity_links(cypher_links, lexical_links, top_k=top_k)

    def evidence_path_expand(
        self,
        semantic_candidates: list[SemanticCandidate],
        entity_links: list[QueryEntityLink],
        plan: RetrievalPlan,
    ) -> list[ContextItem]:
        seed_node_ids = [
            *[candidate.node_id for candidate in semantic_candidates],
            *[link.entity_id for link in entity_links],
        ]
        return self._cypher_expand(seed_node_ids, plan.relation_depth, limit=plan.evidence_budget * 8)

    def relation_expand(self, seed_node_ids: list[str], depth: int) -> list[ContextItem]:
        if self.driver is None:
            return super().relation_expand(seed_node_ids, depth)
        return self._cypher_expand(seed_node_ids, depth, limit=200)

    def _cypher_expand(self, seed_node_ids: list[str], depth: int, limit: int) -> list[ContextItem]:
        if self.driver is None:
            return super().relation_expand(seed_node_ids, depth)
        with self.driver.session() as session:
            rows = session.run(
                f"""
                MATCH path = (seed)-[:HAS_CHUNK|MENTIONS|RELATED_TO|EVIDENCED_BY*0..{depth}]-(chunk:Chunk)
                WHERE seed.node_id IN $seed_node_ids
                RETURN chunk.node_id AS node_id,
                       length(path) AS depth,
                       [n IN nodes(path) | n.node_id] AS node_ids,
                       [r IN relationships(path) | type(r)] AS edge_types
                ORDER BY depth ASC, node_id ASC
                LIMIT $limit
                """,
                seed_node_ids=seed_node_ids,
                limit=limit,
            ).data()
        self.relation_expand_mode = "cypher_planned_paths"
        items: dict[str, ContextItem] = {}
        for row in rows:
            node_id = row["node_id"]
            depth_value = int(row["depth"])
            path = interleave_path(row["node_ids"], row["edge_types"])
            score = 1.0 / max(1, depth_value + 1)
            item = ContextItem(
                node_id=node_id,
                node_type="Chunk",
                score=score,
                reason=f"Neo4j Cypher relation path depth={depth_value}",
                path=path,
            )
            previous = items.get(node_id)
            if previous is None or item.score > previous.score:
                items[node_id] = item
        return sorted(items.values(), key=lambda item: (-item.score, item.node_id))

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()
            self.driver = None

    def _create_lookup_indexes(self, session) -> None:
        statements = [
            "CREATE CONSTRAINT document_node_id IF NOT EXISTS FOR (d:Document) REQUIRE d.node_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_node_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.node_id IS UNIQUE",
            "CREATE CONSTRAINT entity_node_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.node_id IS UNIQUE",
            "CREATE INDEX entity_canonical_name IF NOT EXISTS FOR (e:Entity) ON (e.canonical_name)",
            "CREATE INDEX chunk_document_id IF NOT EXISTS FOR (c:Chunk) ON (c.document_id)",
        ]
        for statement in statements:
            session.run(statement).consume()
        session.run("CALL db.awaitIndexes(30)").consume()

    def _create_vector_indexes(self, session, artifacts: GraphArtifacts) -> None:
        chunk_dimension = next((len(record.vector) for record in artifacts.embeddings if record.owner_type == "Chunk"), 0)
        entity_dimension = next((len(record.vector) for record in artifacts.embeddings if record.owner_type == "Entity"), 0)
        for index_name in (self.vector_index_name, self.entity_vector_index_name):
            session.run(f"DROP INDEX {index_name} IF EXISTS").consume()
        session.run(
            f"""
            CREATE VECTOR INDEX {self.vector_index_name}
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {{indexConfig: {{
              `vector.dimensions`: {chunk_dimension},
              `vector.similarity_function`: 'cosine'
            }}}}
            """
        ).consume()
        if entity_dimension:
            session.run(
                f"""
                CREATE VECTOR INDEX {self.entity_vector_index_name}
                FOR (e:Entity) ON (e.embedding)
                OPTIONS {{indexConfig: {{
                  `vector.dimensions`: {entity_dimension},
                  `vector.similarity_function`: 'cosine'
                }}}}
                """
            ).consume()
        session.run("CALL db.awaitIndex($index_name, 30)", index_name=self.vector_index_name).consume()
        if entity_dimension:
            session.run("CALL db.awaitIndex($index_name, 30)", index_name=self.entity_vector_index_name).consume()

    def _load_nodes(self, session, artifacts: GraphArtifacts) -> None:
        session.run(
            """
            UNWIND $rows AS row
            CREATE (:Document {
              node_id: row.node_id, title: row.title, source_path: row.source_path,
              source_type: row.source_type, text: row.text
            })
            """,
            rows=[document.__dict__ | {"node_id": document.document_id} for document in artifacts.documents],
        ).consume()
        session.run(
            """
            UNWIND $rows AS row
            CREATE (:Chunk {
              node_id: row.chunk_id, document_id: row.document_id, chunk_index: row.chunk_index,
              text: row.text, token_count: row.token_count, embedding: row.embedding
            })
            """,
            rows=[
                chunk.__dict__ | {"embedding": self.embeddings_by_owner[chunk.chunk_id].vector}
                for chunk in artifacts.chunks
            ],
        ).consume()
        session.run(
            """
            UNWIND $rows AS row
            CREATE (:Entity {
              node_id: row.entity_id, name: row.name, canonical_name: row.canonical_name,
              entity_type: row.entity_type, description: row.description, embedding: row.embedding
            })
            """,
            rows=[
                entity.__dict__ | {"embedding": self.embeddings_by_owner[entity.entity_id].vector}
                for entity in artifacts.entities
            ],
        ).consume()

    def _load_edges(self, session, artifacts: GraphArtifacts) -> None:
        session.run(
            """
            UNWIND $rows AS row
            MATCH (d:Document {node_id: row.document_id})
            MATCH (c:Chunk {node_id: row.chunk_id})
            CREATE (d)-[:HAS_CHUNK {weight: 1.0}]->(c)
            """,
            rows=[{"document_id": chunk.document_id, "chunk_id": chunk.chunk_id} for chunk in artifacts.chunks],
        ).consume()
        session.run(
            """
            UNWIND $rows AS row
            MATCH (c:Chunk {node_id: row.chunk_id})
            MATCH (e:Entity {node_id: row.entity_id})
            CREATE (c)-[:MENTIONS {mention_text: row.mention_text, weight: row.confidence}]->(e)
            """,
            rows=[mention.__dict__ for mention in artifacts.mentions],
        ).consume()
        session.run(
            """
            UNWIND $rows AS row
            MATCH (s:Entity {node_id: row.source_entity_id})
            MATCH (t:Entity {node_id: row.target_entity_id})
            MATCH (c:Chunk {node_id: row.evidence_chunk_id})
            CREATE (s)-[:RELATED_TO {
              relationship_id: row.relationship_id, weight: row.weight,
              description: row.description, evidence_chunk_id: row.evidence_chunk_id
            }]->(t)
            CREATE (s)-[:EVIDENCED_BY {relationship_id: row.relationship_id, weight: row.weight}]->(c)
            CREATE (t)-[:EVIDENCED_BY {relationship_id: row.relationship_id, weight: row.weight}]->(c)
            """,
            rows=[relationship.__dict__ for relationship in artifacts.relationships],
        ).consume()


def interleave_path(node_ids: list[str], edge_types: list[str]) -> list[str]:
    if not node_ids:
        return []
    path = [node_ids[0]]
    for edge_type, node_id in zip(edge_types, node_ids[1:], strict=False):
        path.extend([edge_type, node_id])
    return path
