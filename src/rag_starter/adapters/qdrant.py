from __future__ import annotations

from .base import VectorRecord, VectorStoreAdapter


class QdrantVectorStore(VectorStoreAdapter):
    def __init__(self, client, collection_name: str) -> None:
        self.client = client
        self.collection_name = collection_name

    def upsert(self, items: list[VectorRecord]) -> None:
        points = [
            {
                "id": item.id,
                "vector": item.vector,
                "payload": {"text": item.text, **item.metadata},
            }
            for item in items
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def query(self, vector: list[float], top_k: int = 5, filters: dict | None = None) -> list[VectorRecord]:
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=top_k,
            query_filter=filters,
        )
        points = getattr(result, "points", result)
        records: list[VectorRecord] = []
        for point in points:
            payload = dict(getattr(point, "payload", {}) or {})
            text = payload.pop("text", "")
            records.append(
                VectorRecord(
                    id=str(getattr(point, "id")),
                    vector=list(getattr(point, "vector", []) or []),
                    text=text,
                    metadata=payload,
                )
            )
        return records

    def delete(self, ids: list[str]) -> None:
        self.client.delete(collection_name=self.collection_name, points_selector=ids)

    def clear(self) -> None:
        self.client.delete_collection(collection_name=self.collection_name)
