import unittest
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch
from src.vector_store import VectorStore

class TestVectorStoreReset(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.metadata_file = os.path.join(self.test_dir, "metadata.jsonl")
        self.checkpoint_path = self.metadata_file + ".checkpoint"

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch('src.vector_store.chromadb.PersistentClient')
    @patch('src.vector_store.embedding_functions.SentenceTransformerEmbeddingFunction')
    @patch('src.vector_store.os.makedirs')
    def test_reset_db(self, mock_makedirs, mock_emb, mock_client_cls):
        # Setup Mock DB
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Initialize VectorStore
        vs = VectorStore()

        # 1. Simulate existing checkpoint
        with open(self.checkpoint_path, 'w') as f:
            f.write("dummy data")
        self.assertTrue(os.path.exists(self.checkpoint_path))

        # 2. Call reset_db
        print("\n[Test] Calling reset_db...")
        vs.reset_db(self.metadata_file)

        # 3. Verify checkpoint deleted
        self.assertFalse(os.path.exists(self.checkpoint_path), "Checkpoint file should be deleted")

        # 4. Verify collection deleted and recreated
        mock_client.delete_collection.assert_called_with("pubmed_papers")
        self.assertEqual(mock_client.get_or_create_collection.call_count, 2, "Should be called once in init and once in reset")

if __name__ == "__main__":
    unittest.main()
