import unittest
from src.text_processor import clean_text, chunk_text

class TestTextProcessor(unittest.TestCase):

    def test_clean_text(self):
        text = "  This is a   test sentence.  "
        cleaned_text = clean_text(text)
        self.assertEqual(cleaned_text, "This is a test sentence.")

    def test_chunk_text(self):
        text = "This is a long piece of text that needs to be chunked."
        chunks = chunk_text(text, chunk_size=10, overlap=3)
        self.assertEqual(len(chunks), 8)
        self.assertEqual(chunks[0], "This is a ")
        self.assertEqual(chunks[1], " a long pi")

if __name__ == '__main__':
    unittest.main()
