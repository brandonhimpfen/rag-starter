from __future__ import annotations

from .base import VectorRecord, VectorStoreAdapter


class ChromaVectorStore(VectorStoreAdapter):
    def __init__(self, collection) -> None:
        self.collection = collection

    @classmethod
    def from_client(cls, client, name: str = "rag-starter") -> "ChromaVectorStore":
        collection = client.get_or_create_collection(name=name)
        return cls(collection=collection)

    def upsert(self, items: list[VectorRecord]) -> None:
        self.collection.upsert(
            ids=[item.id for item in items],
            embeddings=[item.vector for item in items],
            documents=[item.text for item in items],
            metadatas=[item.metadata for item in items],
        )

    def query(self, vector: list[float], top_k: int = 5, filters: dict | None = None) -> list[VectorRecord]:
        result = self.collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=filters or None,
        )
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        vectors = result.get("embeddings", [[]])[0] if result.get("embeddings") else [[] for _ in ids]
        return [
            VectorRecord(id=item_id, vector=vec, text=text, metadata=meta or {})
            for item_id, vec, text, meta in zip(ids, vectors, docs, metas)
        ]

    def delete(self, ids: list[str]) -> None:
        self.collection.delete(ids=ids)

    def clear(self) -> None:
        all_items = self.collection.get()
        ids = all_items.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)
