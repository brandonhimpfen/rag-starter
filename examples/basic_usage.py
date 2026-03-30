from rag_starter.adapters.inmemory import InMemoryVectorStore
from rag_starter.chunking import chunk_text
from rag_starter.document import Chunk, Document
from rag_starter.embedder import HashingEmbedder
from rag_starter.pipeline import RAGPipeline


def main() -> None:
    doc = Document(
        id="rag-intro",
        text=(
            "Retrieval-augmented generation combines search and generation. "
            "A vector database stores embeddings for chunks. "
            "At query time, the system retrieves relevant chunks and places them into the prompt."
        ),
        metadata={"title": "RAG Intro", "topic": "rag"},
    )

    chunks = [
        Chunk(id=f"rag-intro-{i}", document_id=doc.id, text=text, metadata=doc.metadata)
        for i, text in enumerate(chunk_text(doc.text, chunk_size=80, overlap=15), start=1)
    ]

    pipeline = RAGPipeline(
        store=InMemoryVectorStore(),
        embedder=HashingEmbedder(dimensions=64),
    )
    pipeline.index_chunks(chunks)

    result = pipeline.retrieve("What stores embeddings in a RAG system?", top_k=2)

    print("Top matches:\n")
    for match in result.matches:
        print(f"score={match.score:.4f} | {match.chunk.text}")

    print("\nPrompt preview:\n")
    print(result.to_prompt())


if __name__ == "__main__":
    main()
