import os
import json
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional
from tqdm import tqdm

# Constants
CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db")
COLLECTION_NAME = "pubmed_papers"

class VectorStore:
    def __init__(self):
        """
        Initialize ChromaDB client and collection.
        Uses a local persistent storage.
        """
        # Ensure the data directory exists
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        # Use sentence-transformers for embeddings
        # 'all-MiniLM-L6-v2' is a good balance of speed and quality
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"} # Use cosine similarity
        )

    def index_papers(self, metadata_file: str, batch_size: int = 100):
        """
        Read metadata.jsonl and index papers into ChromaDB.
        
        Args:
            metadata_file: Path to the metadata.jsonl file.
            batch_size: Number of documents to process in a batch.
        """
        if not os.path.exists(metadata_file):
            print(f"Error: Metadata file not found at {metadata_file}")
            return

        print(f"Indexing papers from {metadata_file}...")
        
        documents = []
        metadatas = []
        ids = []
        
        # Count lines first if possible, otherwise just use None
        try:
            total_lines = sum(1 for _ in open(metadata_file, "r", encoding="utf-8"))
        except:
            total_lines = None
            
        with open(metadata_file, "r", encoding="utf-8") as f:
            # Wrap iterator in tqdm
            iterator = tqdm(f, total=total_lines, desc="Indexing")
            for line in iterator:
                try:
                    data = json.loads(line)
                    pmid = data.get("pmid")
                    title = data.get("title", "")
                    abstract = data.get("abstract", "")
                    
                    if not pmid or (not title and not abstract):
                        continue
                        
                    # Prepare document text for embedding (Title + Abstract)
                    doc_text = f"Title: {title}\nAbstract: {abstract}"
                    
                    # Prepare metadata (store essential info for retrieval)
                    meta = {
                        "pmid": pmid,
                        "title": title[:200], # Truncate to save space if needed
                        "journal": data.get("journal", ""),
                        "year": data.get("year", ""),
                    }
                    
                    documents.append(doc_text)
                    metadatas.append(meta)
                    ids.append(pmid)
                    
                    # Batch upsert
                    if len(documents) >= batch_size:
                        self.collection.upsert(
                            documents=documents,
                            metadatas=metadatas,
                            ids=ids
                        )
                        documents = []
                        metadatas = []
                        ids = []
                        
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error processing line: {e}")
                    continue

        # Upsert remaining
        if documents:
            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        
        print(f"Indexing complete. {len(ids)} remaining documents processed.")

    def query(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Perform a semantic search.
        
        Args:
            query: The user's query string.
            limit: Number of results to return.
            
        Returns:
            List of dictionaries containing paper metadata and full text.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=limit
        )
        
        candidates = []
        if not results['ids']:
            return candidates
            
        ids = results['ids'][0]
        metas = results['metadatas'][0]
        documents = results['documents'][0]
        
        for i, doc_id in enumerate(ids):
            meta = metadatas[i]
            doc_text = documents[i]
            
            # Simple parsing of our stored doc format:
            abstract = ""
            if "Abstract: " in doc_text:
                parts = doc_text.split("Abstract: ", 1)
                if len(parts) > 1:
                    abstract = parts[1]
            
            candidate = {
                "pmid": doc_id,
                "title": meta.get("title", ""),
                "journal": meta.get("journal", ""),
                "year": meta.get("year", ""),
                "abstract": abstract,
                "authors": [] # We didn't store authors in vector metadata to save space
            }
            candidates.append(candidate)
            
        return candidates

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        # Alias for query method to match interface
        return self.query(query, limit)

if __name__ == "__main__":
    # Test
    vs = VectorStore()
    # vs.index_papers("data/metadata.jsonl")
    # print(vs.search("cancer treatment"))
    print("VectorStore initialized.")
