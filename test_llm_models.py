from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import json
import time
import argparse
from encode import QueryEncoder
from vector_db import VectorDB
from llm_generation import LLMGenerator


def create_prompt(query: str, retrieved_docs: List[Tuple[int, str]]) -> str:
    """Component 4: Prompt augmentation."""
    prompt = f"{query} Top documents:"
    for doc_id, text in retrieved_docs:
        prompt += f" {text}"
    return prompt


def test_llm_model(model_path: Path, num_queries: int = 10):
    """
    Test a specific LLM model on queries and report timing.
    """
    print("=" * 80)
    print(f"Testing LLM Model: {model_path.name}")
    print("=" * 80)
    
    if not model_path.exists():
        print(f"ERROR: Model file not found at {model_path}")
        return None
    
    # Load queries
    with Path("queries.json").open("r", encoding="utf-8") as f:
        all_queries = json.load(f)
    
    test_queries = all_queries[:num_queries]
    print(f"\nTesting with {len(test_queries)} queries\n")
    
    # Initialize system
    print("Loading vector database...")
    vector_db = VectorDB(Path("preprocessed_documents.json"))
    
    print("Loading query encoder...")
    encoder = QueryEncoder()
    
    print(f"Loading LLM from {model_path}...")
    llm = LLMGenerator(model_path)
    print()
    
    # Process queries
    print("=" * 80)
    print("Running queries...")
    print("=" * 80 + "\n")
    
    encode_times = []
    search_times = []
    prompt_times = []
    llm_times = []
    total_times = []
    
    for i, query_obj in enumerate(test_queries, start=1):
        query_text = query_obj["text"]
        query_id = query_obj["id"]
        
        print(f"Query {i}/{len(test_queries)} (ID: {query_id}): {query_text}")
        
        # Encode
        start = time.time()
        query_embedding = encoder.encode(query_text)
        encode_time = time.time() - start
        encode_times.append(encode_time)
        
        # Search
        start = time.time()
        retrieved_docs = vector_db.search(query_embedding, top_k=3)
        search_time = time.time() - start
        search_times.append(search_time)
        
        # Prompt
        start = time.time()
        augmented_prompt = create_prompt(query_text, retrieved_docs)
        prompt_time = time.time() - start
        prompt_times.append(prompt_time)
        
        # LLM generation
        start = time.time()
        answer = llm.generate(augmented_prompt, max_tokens=256)
        llm_time = time.time() - start
        llm_times.append(llm_time)
        
        total_time = encode_time + search_time + prompt_time + llm_time
        total_times.append(total_time)
        
        print(f"  Encoding: {encode_time*1000:.2f} ms")
        print(f"  Search:   {search_time*1000:.2f} ms")
        print(f"  Prompt:   {prompt_time*1000:.2f} ms")
        print(f"  LLM:      {llm_time*1000:.2f} ms")
        print(f"  Total:    {total_time*1000:.2f} ms")
        print()
    
    # Calculate statistics
    import numpy as np
    
    # Calculate means, standard deviations, and variances
    results = {
        'model_name': model_path.name,
        'model_path': str(model_path),
        'num_queries': len(test_queries),
        'avg_encode': np.mean(encode_times) * 1000,
        'std_encode': np.std(encode_times) * 1000,
        'var_encode': np.var(encode_times) * 1000 * 1000,  # Variance in ms^2
        'avg_search': np.mean(search_times) * 1000,
        'std_search': np.std(search_times) * 1000,
        'var_search': np.var(search_times) * 1000 * 1000,
        'avg_prompt': np.mean(prompt_times) * 1000,
        'std_prompt': np.std(prompt_times) * 1000,
        'var_prompt': np.var(prompt_times) * 1000 * 1000,
        'avg_llm': np.mean(llm_times) * 1000,
        'std_llm': np.std(llm_times) * 1000,
        'var_llm': np.var(llm_times) * 1000 * 1000,
        'avg_total': np.mean(total_times) * 1000,
        'std_total': np.std(total_times) * 1000,
        'var_total': np.var(total_times) * 1000 * 1000,
    }
    
    # Print summary with both std and variance
    print("=" * 80)
    print("Summary Statistics (with Variance)")
    print("=" * 80)
    print(f"Model: {model_path.name}")
    print(f"Number of queries: {len(test_queries)}")
    print()
    print(f"{'Component':<25} {'Mean (ms)':<15} {'Std Dev (ms)':<15} {'Variance (ms²)':<15}")
    print("-" * 80)
    print(f"{'Query Encoding':<25} {results['avg_encode']:<15.2f} {results['std_encode']:<15.2f} {results['var_encode']:<15.2f}")
    print(f"{'Vector Search':<25} {results['avg_search']:<15.2f} {results['std_search']:<15.2f} {results['var_search']:<15.2f}")
    print(f"{'Prompt Augmentation':<25} {results['avg_prompt']:<15.2f} {results['std_prompt']:<15.2f} {results['var_prompt']:<15.2f}")
    print(f"{'LLM Generation':<25} {results['avg_llm']:<15.2f} {results['std_llm']:<15.2f} {results['var_llm']:<15.2f}")
    print(f"{'Total Pipeline':<25} {results['avg_total']:<15.2f} {results['std_total']:<15.2f} {results['var_total']:<15.2f}")
    print("=" * 80 + "\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test different LLM models")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to GGUF model file (e.g., tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf)",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=10,
        help="Number of queries to test (default: 10)",
    )
    
    args = parser.parse_args()
    
    results = test_llm_model(args.model, args.num_queries)
    
    if results:
        print("\nTest completed successfully!")
        print(f"\nTo test another model, run:")
        print(f"  python test_llm_models.py --model <path_to_model.gguf>")
        print(f"\nExample models you could test:")
        print(f"  - TinyLlama (current): tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf")
        print(f"  - Larger models from HuggingFace/TheBloke (if you have them)")


if __name__ == "__main__":
    main()

