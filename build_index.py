import json
import math
import os
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DOCS_PATH = BASE_DIR / "drug_docs.json"
INDEX_DIR = BASE_DIR / "rag_index"
EMBED_MODEL = "all-MiniLM-L6-v2"

# The system uses a chunk size of 400 tokens with an overlap of 80 tokens. 
# Chunk sizes of 256, 400, and 512 tokens were tested. The 400-token configuration provided the best balance between context preservation and retrieval precision and was therefore used in the final system.
def simple_chunk(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    """Split long source text into overlapping chunks for retrieval."""
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def tokenize(text: str) -> List[str]:
    """Lowercase tokenization shared by the BM25 indexer and retriever."""
    return re.findall(r"\b\w+\b", text.lower())


def load_docs(path: Path) -> List[Dict]:
    """Load the raw medication documents from disk."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_chunks(docs: List[Dict]) -> List[Dict]:
    """Normalize source documents into chunk records used by the app."""
    chunks = []
    for doc in docs:
        for i, chunk in enumerate(simple_chunk(doc["document"])):
            chunks.append(
                {
                    "chunk_id": f'{doc["id"]}_chunk{i}',
                    "doc_id": doc["id"],
                    "title": doc["medicine_name"],
                    "source": doc["source_website"],
                    "url": doc.get("source_website", ""),
                    "text": chunk,
                }
            )
    return chunks


def build_bm25_index(chunks: List[Dict]) -> Dict:
    """Precompute BM25 statistics so lexical retrieval is fast at runtime."""
    tokenized_docs = [tokenize(chunk["text"]) for chunk in chunks]
    doc_freqs = []
    doc_lengths = []
    document_frequency = Counter()

    for tokens in tokenized_docs:
        frequencies = Counter(tokens)
        doc_freqs.append(dict(frequencies))
        doc_lengths.append(len(tokens))
        for token in frequencies:
            document_frequency[token] += 1

    total_docs = len(tokenized_docs)
    avgdl = sum(doc_lengths) / total_docs if total_docs else 0.0
    idf = {
        token: math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
        for token, freq in document_frequency.items()
    }

    return {
        "tokenized_docs": tokenized_docs,
        "doc_freqs": doc_freqs,
        "doc_lengths": doc_lengths,
        "avgdl": avgdl,
        "idf": idf,
        "k1": 1.5,
        "b": 0.75,
    }


def main() -> None:
    """Build the dense FAISS index and the lexical BM25 index together."""
    os.makedirs(INDEX_DIR, exist_ok=True)

    docs = load_docs(DOCS_PATH)
    chunks = build_chunks(docs)

    model = SentenceTransformer(EMBED_MODEL)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    dim = embeddings.shape[1]
    dense_index = faiss.IndexFlatIP(dim)
    dense_index.add(embeddings.astype(np.float32))

    # BM25 complements dense retrieval by rewarding exact token overlap.
    bm25_index = build_bm25_index(chunks)

    faiss.write_index(dense_index, str(INDEX_DIR / "docs.index"))
    with open(INDEX_DIR / "chunks.pkl", "wb") as file:
        pickle.dump(chunks, file)
    with open(INDEX_DIR / "bm25.pkl", "wb") as file:
        pickle.dump(bm25_index, file)

    print(f"Built hybrid index with {len(chunks)} chunks.")


if __name__ == "__main__":
    main()
