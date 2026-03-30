import unittest

from rag_starter.chunking import chunk_text


class ChunkingTests(unittest.TestCase):
    def test_chunk_text_returns_chunks(self):
        text = "One two three four five six seven eight nine ten"
        chunks = chunk_text(text, chunk_size=15, overlap=3)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(chunks))

    def test_chunk_text_validates_arguments(self):
        with self.assertRaises(ValueError):
            chunk_text("hello", chunk_size=0)
        with self.assertRaises(ValueError):
            chunk_text("hello", chunk_size=10, overlap=10)


if __name__ == "__main__":
    unittest.main()
