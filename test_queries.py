from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import json
import time
from encode import QueryEncoder
from vector_db import VectorDB
from llm_generation import LLMGenerator


def create_prompt(query: str, retrieved_docs: List[Tuple[int, str]]) -> str:
    """
    Component 4: Prompt augmentation.
    Combines user query with retrieved documents into a structured prompt.
    """
    prompt = f"{query} Top documents:"
    for doc_id, text in retrieved_docs:
        prompt += f" {text}"
    return prompt


def main():
    """
    Test the current model on the first 10 queries from queries.json
    """
    print("=" * 80)
    print("Testing RAG System on First 10 Queries")
    print("=" * 80)
    
    # 1. Load queries
    queries_path = Path("queries.json")
    if not queries_path.exists():
        print(f"ERROR: {queries_path} not found!")
        return
    
    with queries_path.open("r", encoding="utf-8") as f:
        all_queries = json.load(f)
    
    # Get first 10 queries
    test_queries = all_queries[:10]
    print(f"\nLoaded {len(test_queries)} queries for testing\n")
    
    # 2. Initialize system
    embeddings_path = Path("preprocessed_documents.json")
    if not embeddings_path.exists():
        print(f"ERROR: {embeddings_path} not found!")
        print("Please run data_preprocess.py first to generate preprocessed_documents.json")
        return
    
    print("Loading vector database...")
    vector_db = VectorDB(embeddings_path)
    
    print("Loading query encoder...")
    encoder = QueryEncoder()
    
    model_path = Path("tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf")
    if not model_path.exists():
        print(f"\nERROR: Model file not found at {model_path}")
        return
    
    print("Loading LLM...")
    llm = LLMGenerator(model_path)
    
    print("\n" + "=" * 80)
    print("Running queries...")
    print("=" * 80 + "\n")
    
    # 3. Process each query
    results = []
    for i, query_obj in enumerate(test_queries, start=1):
        query_text = query_obj["text"]
        query_id = query_obj["id"]
        
        print(f"Query {i}/10 (ID: {query_id}): {query_text}")
        print("-" * 80)
        
        # Execute RAG pipeline with timing
        start_time = time.time()
        query_embedding = encoder.encode(query_text)
        encode_time = time.time() - start_time
        
        start_time = time.time()
        retrieved_docs = vector_db.search(query_embedding, top_k=3)
        search_time = time.time() - start_time
        
        start_time = time.time()
        augmented_prompt = create_prompt(query_text, retrieved_docs)
        prompt_time = time.time() - start_time
        
        start_time = time.time()
        answer = llm.generate(augmented_prompt, max_tokens=256)
        llm_time = time.time() - start_time
        
        total_time = encode_time + search_time + prompt_time + llm_time
        
        # Store results
        results.append({
            "query_id": query_id,
            "query_text": query_text,
            "encode_time": encode_time,
            "search_time": search_time,
            "prompt_time": prompt_time,
            "llm_time": llm_time,
            "total_time": total_time,
            "answer": answer
        })
        
        # Print timing
        print(f"  Encoding:        {encode_time*1000:.2f} ms")
        print(f"  Vector Search:   {search_time*1000:.2f} ms")
        print(f"  Prompt Aug:      {prompt_time*1000:.2f} ms")
        print(f"  LLM Generation:  {llm_time*1000:.2f} ms")
        print(f"  Total:           {total_time*1000:.2f} ms ({total_time:.3f} s)")
        print()
    
    # 4. Print summary
    print("=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    
    avg_encode = sum(r["encode_time"] for r in results) / len(results) * 1000
    avg_search = sum(r["search_time"] for r in results) / len(results) * 1000
    avg_prompt = sum(r["prompt_time"] for r in results) / len(results) * 1000
    avg_llm = sum(r["llm_time"] for r in results) / len(results) * 1000
    avg_total = sum(r["total_time"] for r in results) / len(results) * 1000
    
    print(f"Average Query Encoding:      {avg_encode:.2f} ms")
    print(f"Average Vector Search:       {avg_search:.2f} ms")
    print(f"Average Prompt Augmentation: {avg_prompt:.2f} ms")
    print(f"Average LLM Generation:      {avg_llm:.2f} ms")
    print(f"Average Total Pipeline:     {avg_total:.2f} ms")
    print("=" * 80)


if __name__ == "__main__":
    main()

