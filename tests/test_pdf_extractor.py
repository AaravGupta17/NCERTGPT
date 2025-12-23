import unittest
import os
from src.pdf_extractor import extract_text_from_pdf

class TestPdfExtractor(unittest.TestCase):

    def setUp(self):
        self.pdf_path = "tests/test.pdf"

    def test_extract_text_from_pdf(self):
        text = extract_text_from_pdf(self.pdf_path)
        self.assertIn("This is a test PDF file for the NCERT chatbot.", text.strip())

if __name__ == '__main__':
    unittest.main()
