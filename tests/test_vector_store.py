import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vector_store import VectorStore

class TestVectorStore(unittest.TestCase):
    @patch('src.vector_store.chromadb.PersistentClient')
    @patch('src.vector_store.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_query_bug(self, mock_embedding, mock_client):
        # Setup mock collection
        mock_collection = MagicMock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        # Initialize VectorStore (will use mocks)
        vs = VectorStore()
        # Ensure collection is set to our mock (it should be via get_or_create_collection mock)
        vs.collection = mock_collection

        # Setup mock query return value
        # Structure matching ChromaDB query result: {'ids': [['id1']], 'metadatas': [[{...}]], 'documents': [['...']]}
        # Note: ChromaDB returns lists of lists because it supports batch queries.
        mock_collection.query.return_value = {
            'ids': [['12345']],
            'metadatas': [[{'title': 'Test Title', 'journal': 'Test Journal', 'year': '2023'}]],
            'documents': [['Title: Test Title\nAbstract: Test Abstract']]
        }

        # Run query
        # This is expected to fail with NameError due to 'metadatas' typo in src/vector_store.py
        vs.query("test query")

if __name__ == '__main__':
    unittest.main()
