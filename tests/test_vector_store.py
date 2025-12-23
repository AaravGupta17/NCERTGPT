import unittest
import os
from src.vector_store import VectorStore

class TestVectorStore(unittest.TestCase):

    def setUp(self):
        self.vector_store = VectorStore()
        self.texts = ["This is the first sentence.", "This is the second sentence."]
        self.vector_store.add_texts(self.texts)

    def test_add_texts(self):
        self.assertEqual(len(self.vector_store.documents), 2)
        self.assertIsNotNone(self.vector_store.index)

    def test_search(self):
        results = self.vector_store.search("first sentence", k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], "This is the first sentence.")

    def test_save_and_load_index(self):
        index_path = "test_index.faiss"
        documents_path = "test_documents.txt"
        self.vector_store.save_index(index_path, documents_path)

        new_vector_store = VectorStore()
        new_vector_store.load_index(index_path, documents_path)

        self.assertEqual(len(new_vector_store.documents), 2)
        results = new_vector_store.search("second sentence", k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], "This is the second sentence.")

        os.remove(index_path)
        os.remove(documents_path)


if __name__ == '__main__':
    unittest.main()
