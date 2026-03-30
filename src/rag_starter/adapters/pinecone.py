from __future__ import annotations

from .base import VectorRecord, VectorStoreAdapter


class PineconeVectorStore(VectorStoreAdapter):
    def __init__(self, index) -> None:
        self.index = index

    def upsert(self, items: list[VectorRecord]) -> None:
        vectors = [
            {
                "id": item.id,
                "values": item.vector,
                "metadata": {"text": item.text, **item.metadata},
            }
            for item in items
        ]
        self.index.upsert(vectors=vectors)

    def query(self, vector: list[float], top_k: int = 5, filters: dict | None = None) -> list[VectorRecord]:
        result = self.index.query(vector=vector, top_k=top_k, filter=filters, include_metadata=True)
        matches = result.get("matches", []) if isinstance(result, dict) else getattr(result, "matches", [])
        records: list[VectorRecord] = []
        for match in matches:
            metadata = dict(getattr(match, "metadata", None) or match.get("metadata", {}) or {})
            text = metadata.pop("text", "")
            records.append(
                VectorRecord(
                    id=str(getattr(match, "id", None) or match.get("id")),
                    vector=list(getattr(match, "values", None) or match.get("values", []) or []),
                    text=text,
                    metadata=metadata,
                )
            )
        return records

    def delete(self, ids: list[str]) -> None:
        self.index.delete(ids=ids)

    def clear(self) -> None:
        self.index.delete(delete_all=True)
