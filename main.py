from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import time
from encode import QueryEncoder
from vector_db import VectorDB
from llm_generation import LLMGenerator


def create_prompt(query: str, retrieved_docs: List[Tuple[int, str]]) -> str:
    """
    Component 4: Prompt augmentation.
    Combines user query with retrieved documents into a structured prompt.
    
    Format: query + " Top documents:" + top1_doc_text + top2_doc_text + top3_doc_text
    """
    prompt = f"{query} Top documents:"
    for doc_id, text in retrieved_docs:
        prompt += f" {text}"
    return prompt


def main():
    """
    Component 6: Main interactive RAG system.
    Orchestrates the complete RAG pipeline.
    """
    print("=" * 80)
    print("RAG System Initialization")
    print("=" * 80)
    
    # 1. Initialize system by loading preprocessed documents
    embeddings_path = Path("preprocessed_documents.json")
    if not embeddings_path.exists():
        print(f"ERROR: {embeddings_path} not found!")
        print("Please run data_preprocess.py first to generate preprocessed_documents.json")
        return
    
    vector_db = VectorDB(embeddings_path)
    
    # 2. Load query encoder
    print("\nLoading query encoder...")
    encoder = QueryEncoder()
    
    # 3. Load LLM
    model_path = Path("tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf")
    if not model_path.exists():
        print(f"\nERROR: Model file not found at {model_path}")
        print("Please download it using:")
        print("wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf")
        return
    
    llm = LLMGenerator(model_path)
    
    print("\n" + "=" * 80)
    print("RAG System Ready! Type your questions (or 'quit' to exit).")
    print("=" * 80 + "\n")
    
    # 4. Interactive loop
    while True:
        query = input("Query: ").strip()
        
        if query.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        # Execute RAG pipeline with timing:
        # a) Encode query into embedding
        start_time = time.time()
        query_embedding = encoder.encode(query)
        encode_time = time.time() - start_time
        
        # b) Retrieve top-k documents using vector search
        start_time = time.time()
        retrieved_docs = vector_db.search(query_embedding, top_k=3)
        search_time = time.time() - start_time
        
        # Print the retrieved documents
        print("\nRetrieved documents:")
        print("-" * 80)
        for i, (doc_id, text) in enumerate(retrieved_docs, start=1):
            snippet = text[:200] + ("..." if len(text) > 200 else "")
            print(f"{i}. [ID: {doc_id}] {snippet}")
        print("-" * 80)
        
        # c) Construct context-augmented prompt
        start_time = time.time()
        augmented_prompt = create_prompt(query, retrieved_docs)
        prompt_time = time.time() - start_time
        
        # d) Generate response using LLM
        start_time = time.time()
        answer = llm.generate(augmented_prompt, max_tokens=256)
        llm_time = time.time() - start_time
        
        # e) Display results
        print(f"\nAnswer:\n{answer}\n")
        
        # Print timing breakdown after the answer
        print("=" * 80)
        print("Performance Timing Breakdown")
        print("=" * 80)
        print(f"1. Query Encoding:        {encode_time*1000:.2f} ms")
        print(f"2. Vector Search:         {search_time*1000:.2f} ms")
        print(f"3. Prompt Augmentation:   {prompt_time*1000:.2f} ms")
        print(f"4. LLM Generation:        {llm_time*1000:.2f} ms")
        # Calculate total time
        total_time = encode_time + search_time + prompt_time + llm_time
        print("-" * 80)
        print(f"Total Pipeline Time:      {total_time*1000:.2f} ms ({total_time:.3f} s)")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
