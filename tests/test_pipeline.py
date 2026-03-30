import unittest

from rag_starter.adapters.inmemory import InMemoryVectorStore
from rag_starter.document import Chunk
from rag_starter.embedder import HashingEmbedder
from rag_starter.pipeline import RAGPipeline


class PipelineTests(unittest.TestCase):
    def test_pipeline_indexes_and_retrieves(self):
        pipeline = RAGPipeline(store=InMemoryVectorStore(), embedder=HashingEmbedder(dimensions=32))
        chunks = [
            Chunk(id="1", document_id="doc1", text="A vector database stores embeddings.", metadata={"topic": "rag"}),
            Chunk(id="2", document_id="doc1", text="Chunking splits long text into smaller pieces.", metadata={"topic": "rag"}),
        ]
        pipeline.index_chunks(chunks)

        result = pipeline.retrieve("What stores embeddings?", top_k=1)
        self.assertEqual(len(result.matches), 1)
        self.assertIn("embeddings", result.matches[0].chunk.text.lower())

    def test_pipeline_filtering(self):
        pipeline = RAGPipeline(store=InMemoryVectorStore(), embedder=HashingEmbedder(dimensions=32))
        chunks = [
            Chunk(id="1", document_id="doc1", text="Travel insurance protects travelers.", metadata={"topic": "travel"}),
            Chunk(id="2", document_id="doc2", text="Vector databases help retrieval systems.", metadata={"topic": "rag"}),
        ]
        pipeline.index_chunks(chunks)

        result = pipeline.retrieve("What helps retrieval systems?", top_k=2, filters={"topic": "rag"})
        self.assertEqual(len(result.matches), 1)
        self.assertEqual(result.matches[0].chunk.metadata["topic"], "rag")


if __name__ == "__main__":
    unittest.main()
