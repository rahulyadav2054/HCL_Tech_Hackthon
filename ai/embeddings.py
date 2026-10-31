"""Shared sentence-transformers embedder (bge-small-en-v1.5, 384-dim, normalized)."""
from __future__ import annotations

import os
import threading
from functools import lru_cache

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from ai import config

_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL, device="cpu")


def embed_docs(texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    """Embed documents (no instruction prefix). Returns float32, L2-normalized."""
    with _lock:
        vecs = get_model().encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                  show_progress_bar=show_progress, convert_to_numpy=True)
    return vecs.astype("float32")


def embed_queries(texts: list[str]) -> np.ndarray:
    """Embed retrieval queries with the model's query instruction (bge convention)."""
    return embed_docs([config.EMBED_QUERY_INSTRUCTION + t for t in texts])
