from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Encode MS MARCO passages with BGE.")
    parser.add_argument(
        "--documents",
        type=Path,
        default=Path("documents.json"),
        help="Path to the raw MS MARCO subset JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("preprocessed_documents.json"),
        help="Destination for the encoded JSON output.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of passages to encode per batch.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for SentenceTransformer (cpu or cuda).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="SentenceTransformer model name to use for encoding.",
    )
    return parser.parse_args()


def load_documents(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def encode_documents(documents, model_name = "BAAI/bge-large-en-v1.5",
    batch_size = 64, device = "cpu"):
    model = SentenceTransformer(model_name, device=device)
    texts = [doc["text"] for doc in documents]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,
    )
    return embeddings.astype(np.float32)


def write_output(documents, embeddings: np.ndarray, path: Path):
    assert len(documents) == len(embeddings), "Embedding count mismatch."
    serializable = []
    for doc, emb in zip(documents, embeddings):
        serializable.append(
            {
                "id": doc["id"],
                "text": doc["text"],
                "embedding": emb.tolist(),
            }
        )

    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    print(f"Loading documents from {args.documents} ...")
    documents = load_documents(args.documents)
    print(f"Loaded {len(documents)} passages.")

    print(f"Encoding with {args.model} ...")
    embeddings = encode_documents(
        documents,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
    )

    print(f"Writing {len(documents)} encoded entries to {args.output} ...")
    write_output(documents, embeddings, args.output)
    print("Done.")


if __name__ == "__main__":
    main()

