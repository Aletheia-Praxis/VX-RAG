"""
Tests for ingestion module.
"""

import unittest
from pathlib import Path
# from src.rag.ingest import load_documents, preprocess_documents

class TestIngest(unittest.TestCase):
    """
    Test cases for document ingestion.
    """

    def test_load_documents(self):
        """
        Test loading documents from directory.
        """
        # TODO: Implement test
        # data_dir = Path("test_data")
        # docs = load_documents(data_dir)
        # self.assertIsInstance(docs, list)
        pass

    def test_preprocess_documents(self):
        """
        Test preprocessing documents.
        """
        # TODO: Implement test
        # raw_docs = ["Raw document 1", "Raw document 2"]
        # processed = preprocess_documents(raw_docs)
        # self.assertEqual(len(processed), 2)
        pass

if __name__ == "__main__":
    unittest.main()