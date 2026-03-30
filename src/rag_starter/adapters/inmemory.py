from __future__ import annotations

from .base import VectorRecord, VectorStoreAdapter
from ..utils import cosine_similarity


class InMemoryVectorStore(VectorStoreAdapter):
    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}

    def upsert(self, items: list[VectorRecord]) -> None:
        for item in items:
            self._records[item.id] = item

    def query(
        self,
        vector: list[float],
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[VectorRecord]:
        records = list(self._records.values())
        if filters:
            records = [r for r in records if _matches_filters(r.metadata, filters)]

        ranked = sorted(
            records,
            key=lambda item: cosine_similarity(vector, item.vector),
            reverse=True,
        )
        return ranked[:top_k]

    def delete(self, ids: list[str]) -> None:
        for item_id in ids:
            self._records.pop(item_id, None)

    def clear(self) -> None:
        self._records.clear()



def _matches_filters(metadata: dict, filters: dict) -> bool:
    for key, expected in filters.items():
        if metadata.get(key) != expected:
            return False
    return True
