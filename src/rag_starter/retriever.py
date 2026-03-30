from __future__ import annotations

from .adapters.base import VectorStoreAdapter
from .document import Chunk, RetrievedChunk
from .embedder import Embedder


class Retriever:
    def __init__(self, store: VectorStoreAdapter, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        query_vector = self.embedder.embed([query])[0]
        records = self.store.query(vector=query_vector, top_k=top_k, filters=filters)
        query_vectors = self.embedder.embed([query])
        scored = []
        for record in records:
            chunk = Chunk(
                id=record.id,
                document_id=record.metadata.get("document_id", record.id),
                text=record.text,
                metadata=record.metadata,
            )
            score = _score(query_vectors[0], record.vector)
            scored.append(RetrievedChunk(chunk=chunk, score=score))
        return scored



def _score(a: list[float], b: list[float]) -> float:
    numerator = sum(x * y for x, y in zip(a, b))
    a_norm = sum(x * x for x in a) ** 0.5
    b_norm = sum(y * y for y in b) ** 0.5
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return numerator / (a_norm * b_norm)
