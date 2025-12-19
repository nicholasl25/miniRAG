from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Tuple
import faiss
import numpy as np


def load_embeddings(path: Path):
    """Load embeddings, IDs, and texts from preprocessed_documents.json."""
    with path.open("r", encoding="utf-8") as f:
        entries = json.load(f)
    
    embeddings = np.array(
        [entry["embedding"] for entry in entries],
        dtype=np.float32,
    )
    doc_ids = [entry["id"] for entry in entries]
    doc_texts = [entry["text"] for entry in entries]
    return embeddings, doc_ids, doc_texts


def build_index(embeddings: np.ndarray):
    """Build FAISS IndexFlatL2 from embeddings."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


class VectorDB:
    """
    Component 2 & 3: Vector search and document retrieval.
    Loads preprocessed documents, builds FAISS index, and provides search + retrieval.
    """
    
    def __init__(self, embeddings_path: Path):
        print(f"Loading embeddings from {embeddings_path}...")
        embeddings, doc_ids, doc_texts = load_embeddings(embeddings_path)
        
        print("Building FAISS index...")
        self.index = build_index(embeddings)
        
        # Component 3: In-memory document retrieval
        self.doc_ids = doc_ids
        self.doc_texts = doc_texts
        self.id_to_text = {doc_id: text for doc_id, text in zip(doc_ids, doc_texts)}
        
        print(f"Loaded {len(embeddings)} documents. Index ready.")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[int, str]]:
        """
        Component 2: Search for top-k documents.
        
        Args:
            query_embedding: Query vector of shape (768,)
            top_k: Number of documents to retrieve (default 3)
            
        Returns:
            List of (doc_id, text) tuples for the top-k documents
        """
        query_vec = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_vec, top_k)
        
        # Component 3: Retrieve actual document texts
        results = []
        for idx in indices[0]:
            doc_id = self.doc_ids[idx]
            text = self.doc_texts[idx]
            results.append((doc_id, text))
        
        return results


def main():
    """CLI for testing vector search (Part 1 functionality)."""
    parser = argparse.ArgumentParser(description="FAISS vector search helper.")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("preprocessed_documents.json"),
        help="Path to preprocessed_documents.json",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of nearest neighbors",
    )
    parser.add_argument(
        "--query-id",
        type=int,
        default=0,
        help="Document ID to use as query (for testing)",
    )
    args = parser.parse_args()
    
    db = VectorDB(args.embeddings)
    
    if args.query_id not in db.doc_ids:
        raise ValueError(f"Query id {args.query_id} not found")
    
    query_pos = db.doc_ids.index(args.query_id)
    query_vec = np.array([db.index.reconstruct(query_pos)])
    
    print(f"\nSearching top-{args.k} neighbors for id={args.query_id}...")
    results = db.search(query_vec[0], top_k=args.k)
    
    for rank, (doc_id, text) in enumerate(results, start=1):
        marker = "<-- query" if doc_id == args.query_id else ""
        snippet = text[:120].replace("\n", " ") + ("..." if len(text) > 120 else "")
        print(f"{rank:02d}. id={doc_id:5d} {marker}")
        print(f"    {snippet}\n")


if __name__ == "__main__":
    main()
