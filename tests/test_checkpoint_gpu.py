import unittest
import os
import json
import shutil
import tempfile
from unittest.mock import MagicMock, patch
from src.vector_store import VectorStore

class TestVectorStoreCheckpoint(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.metadata_file = os.path.join(self.test_dir, "metadata.jsonl")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch('src.vector_store.chromadb.PersistentClient')
    @patch('src.vector_store.embedding_functions.SentenceTransformerEmbeddingFunction')
    @patch('src.vector_store.os.makedirs')
    def test_incremental_indexing(self, mock_makedirs, mock_emb, mock_client_cls):
        # Setup Mock DB
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection

        # Initialize VectorStore
        vs = VectorStore()
        # Ensure we are using our mock collection
        vs.collection = mock_collection

        # 1. Create initial data (4 records)
        with open(self.metadata_file, 'w') as f:
            for i in range(4):
                f.write(json.dumps({"pmid": str(i), "title": f"Title {i}", "abstract": f"Abs {i}"}) + "\n")

        # 2. Run Indexing (Batch size 2)
        print("\n[Test] Running initial indexing (4 items)...")
        vs.index_papers(self.metadata_file, batch_size=2)

        # Verify Upserts (should be 2 calls: [0,1], [2,3])
        self.assertEqual(mock_collection.upsert.call_count, 2, "Should upsert 2 batches for initial 4 items")

        # Verify Checkpoint content
        checkpoint_path = self.metadata_file + ".checkpoint"
        self.assertTrue(os.path.exists(checkpoint_path), "Checkpoint file should exist")
        with open(checkpoint_path, 'r') as f:
            cp = json.load(f)
            self.assertEqual(cp['processed_lines'], 4, "Checkpoint should record 4 processed lines")

        # 3. Append data (6 records)
        print("[Test] Appending 6 items...")
        with open(self.metadata_file, 'a') as f:
            for i in range(4, 10):
                f.write(json.dumps({"pmid": str(i), "title": f"Title {i}", "abstract": f"Abs {i}"}) + "\n")

        # Reset mock counts
        mock_collection.upsert.reset_mock()

        # 4. Run Indexing Again
        print("[Test] Resuming indexing...")
        vs.index_papers(self.metadata_file, batch_size=2)

        # Verify Upserts
        # Should process items 4,5 (1 batch), 6,7 (1 batch), 8,9 (1 batch) -> 3 calls
        # If it restarted from 0, it would process 0-9 (5 batches)
        self.assertEqual(mock_collection.upsert.call_count, 3, "Should upsert only 3 batches (6 new items)")

        # Verify final checkpoint
        with open(checkpoint_path, 'r') as f:
            cp = json.load(f)
            self.assertEqual(cp['processed_lines'], 10, "Checkpoint should record 10 processed lines")

if __name__ == "__main__":
    unittest.main()
