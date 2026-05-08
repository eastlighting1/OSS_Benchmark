from ..models import GraphArtifacts, QueryEntityLink
from .base import merge_query_entity_links
from .external_semantic_index import ExternalSemanticIndex
from .neo4j_adapter import Neo4jStorageAdapter


class Neo4jExternalSemanticAdapter(Neo4jStorageAdapter):
    config_id = "neo4j-external-semantic"
    config_name = "Neo4j + external semantic index"
    graph_db = "neo4j"
    vector_db = "chroma"

    def __init__(self, vector_store_dir=None) -> None:
        super().__init__()
        self.external_index = ExternalSemanticIndex(None if vector_store_dir is None else vector_store_dir / "neo4j_chroma")

    def load(self, artifacts: GraphArtifacts) -> None:
        super().load(artifacts)
        self.external_index.build(artifacts.embeddings)
        self.vector_db = self.external_index.name
        self.sync_notes = f"{self.external_index.name} built from Neo4j graph node embeddings"

    def semantic_entry(self, question: str, query_embedding: list[float], top_k: int):
        self.semantic_entry_mode = self.external_index.name
        return self.external_index.search(query_embedding, top_k, owner_type="Chunk")

    def semantic_reentry(self, candidates):
        self.semantic_reentry_mode = "external_hits_to_graph_candidates"
        self.cross_store_join_count += len(candidates)
        return candidates

    def link_query_entities(
        self,
        question: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[QueryEntityLink]:
        external_candidates = self.external_index.search(query_embedding, top_k, owner_type="Entity")
        external_links: list[QueryEntityLink] = []
        for index, candidate in enumerate(external_candidates):
            entity = self.entities_by_id.get(candidate.node_id)
            if entity is None:
                continue
            external_links.append(
                QueryEntityLink(
                    entity_id=entity.entity_id,
                    name=entity.name,
                    score=0.35 + candidate.score,
                    rank=index + 1,
                    matched_text="external_entity_embedding",
                    source=f"{self.external_index.name}_entity_reentry",
                )
            )
        native_links = super().link_query_entities(question, query_embedding, top_k)
        self.cross_store_join_count += len(external_links)
        return merge_query_entity_links(external_links, native_links, top_k=top_k)
