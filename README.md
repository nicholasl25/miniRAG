# miniRAG

A mini project to learn Retrieval-Augmented Generation (RAG) and experiment with different model configurations.

## Models Used

- **Embedding Model:** `BAAI/bge-base-en-v1.5` — Converts text into 768-dimensional vectors for semantic search
- **LLM:** `TinyLlama-1.1B-Chat-v0.3` — Lightweight language model for generating responses

## How It Works

```
User Query → Encode Query → Vector Search → Retrieve Documents → Augment Prompt → LLM Generation → Response
```

1. **Encode Query** — Your question is converted into a vector embedding
2. **Vector Search** — FAISS finds the top 3 most similar documents from the preprocessed corpus
3. **Retrieve Documents** — The actual text of matched documents is fetched
4. **Augment Prompt** — Your query is combined with the retrieved context
5. **LLM Generation** — TinyLlama generates an answer grounded in the retrieved documents

## Setup Instructions

### 1. Install Dependencies

```bash
pip install faiss-cpu llama-cpp-python sentence-transformers tqdm
```

### 2. Download the TinyLlama Model

```bash
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf
```

### 3. Generate Preprocessed Documents

This step encodes all documents into vector embeddings using the BGE model. The output `preprocessed_documents.json` is used as context for the TinyLlama LLM.

```bash
python data_preprocess.py
```

This will:
- Read `documents.json` (the raw MS MARCO dataset)
- Encode each document using BGE
- Output `preprocessed_documents.json` with text + embeddings

**Note:** This takes a few minutes on CPU.

### 4. Run the RAG System

```bash
python main.py
```

Type your questions and get AI-generated answers based on the document corpus!

## Project Structure

```
├── main.py                 # Interactive RAG pipeline
├── data_preprocess.py      # Encodes documents into embeddings
├── encode.py               # Query encoder module
├── vector_db.py            # FAISS vector database
├── llm_generation.py       # TinyLlama wrapper
├── documents.json          # Raw document corpus
└── queries.json            # Sample queries for testing
```
