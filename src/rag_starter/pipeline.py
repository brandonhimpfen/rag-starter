from __future__ import annotations

from dataclasses import dataclass

from .adapters.base import VectorRecord, VectorStoreAdapter
from .document import Chunk, RetrievedChunk
from .embedder import Embedder
from .prompts import build_rag_prompt
from .retriever import Retriever


@dataclass(slots=True)
class RetrievalResult:
    query: str
    matches: list[RetrievedChunk]

    def to_prompt(self) -> str:
        return build_rag_prompt(self.query, self.matches)


class RAGPipeline:
    def __init__(self, store: VectorStoreAdapter, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder
        self.retriever = Retriever(store=store, embedder=embedder)

    def index_chunks(self, chunks: list[Chunk]) -> None:
        texts = [chunk.text for chunk in chunks]
        vectors = self.embedder.embed(texts)
        records = [
            VectorRecord(
                id=chunk.id,
                vector=vector,
                text=chunk.text,
                metadata={"document_id": chunk.document_id, **chunk.metadata},
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self.store.upsert(records)

    def retrieve(self, query: str, top_k: int = 5, filters: dict | None = None) -> RetrievalResult:
        matches = self.retriever.retrieve(query=query, top_k=top_k, filters=filters)
        return RetrievalResult(query=query, matches=matches)
