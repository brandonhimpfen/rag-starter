# rag-starter

[![GitHub Sponsor](https://srv-cdn.himpfen.io/badges/github/github-flat.svg)](https://github.com/sponsors/brandonhimpfen) &nbsp; 
[![Buy Me a Coffee](https://srv-cdn.himpfen.io/badges/buymeacoffee/buymeacoffee-flat.svg)](https://buymeacoffee.com/brandonhimpfen) &nbsp; 
[![Ko-Fi](https://srv-cdn.himpfen.io/badges/kofi/kofi-flat.svg)](https://ko-fi.com/brandonhimpfen) &nbsp; 
[![PayPal](https://srv-cdn.himpfen.io/badges/paypal/paypal-flat.svg)](https://paypal.me/brandonhimpfen)

RAG boilerplate with vector DB adapters.

`rag-starter` is a lightweight Python starter for retrieval-augmented generation workflows. It gives you a small but clean foundation for:

- chunking documents.
- generating embeddings.
- indexing vectors through adapter classes.
- retrieving relevant context for prompts.
- swapping vector backends without rewriting your pipeline.

This repo is intentionally minimal. It is designed as a starter, not a full framework.

## Features

- Small, readable Python package structure.
- Adapter interface for vector stores.
- In-memory adapter included for local development and tests.
- Optional adapter stubs for Chroma, Qdrant, and Pinecone.
- Simple hashing embedder for demos and bootstrapping.
- Retriever and RAG pipeline helpers.
- Example script and tests.

## Project structure

```text
rag-starter/
├── examples/
│   └── basic_usage.py
├── src/
│   └── rag_starter/
│       ├── adapters/
│       │   ├── base.py
│       │   ├── chroma.py
│       │   ├── inmemory.py
│       │   ├── pinecone.py
│       │   └── qdrant.py
│       ├── chunking.py
│       ├── document.py
│       ├── embedder.py
│       ├── pipeline.py
│       ├── prompts.py
│       ├── retriever.py
│       └── utils.py
├── tests/
│   ├── test_chunking.py
│   └── test_pipeline.py
├── pyproject.toml
└── README.md
```

## Install

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e .[chroma]
pip install -e .[qdrant]
pip install -e .[pinecone]
```

## Quick start

```python
from rag_starter.adapters.inmemory import InMemoryVectorStore
from rag_starter.chunking import chunk_text
from rag_starter.document import Document, Chunk
from rag_starter.embedder import HashingEmbedder
from rag_starter.pipeline import RAGPipeline

source = Document(
    id="doc-1",
    text="RAG combines retrieval with generation. Vector databases help store embeddings.",
    metadata={"title": "RAG Notes"},
)

chunks = [
    Chunk(id=f"chunk-{i}", document_id=source.id, text=text, metadata=source.metadata)
    for i, text in enumerate(chunk_text(source.text, chunk_size=60, overlap=10), start=1)
]

embedder = HashingEmbedder(dimensions=64)
store = InMemoryVectorStore()
pipeline = RAGPipeline(store=store, embedder=embedder)

pipeline.index_chunks(chunks)
result = pipeline.retrieve("What helps store embeddings?", top_k=2)

for item in result.matches:
    print(item.score, item.chunk.text)
```

## Adapter model

All vector database backends follow the same interface defined in `VectorStoreAdapter`.

Core methods:

- `upsert(items)`
- `query(vector, top_k, filters=None)`
- `delete(ids)`
- `clear()`

The included `InMemoryVectorStore` is useful for:

- local development.
- tests.
- learning the architecture.
- quickly bootstrapping a prototype.

The optional adapters are intentionally thin wrappers so you can extend them to fit your preferred backend configuration.

## What this starter does not try to do

This starter does not include:

- model serving.
- background ingestion workers.
- file loaders for every format.
- advanced ranking pipelines.
- production auth and tenancy layers.

Those are highly project-specific and are better layered on once your retrieval path is clear.

## Development

Run tests:

```bash
python -m unittest discover -s tests -v
```

## License

MIT
