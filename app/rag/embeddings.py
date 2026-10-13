import logging
import threading
from typing import Sequence
import numpy as np
from sentence_transformers import SentenceTransformer
from app.core.config import get_settings

logger = logging.getLogger(__name__)

_model_lock = threading.Lock()
_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """
    Thread-safe singleton loader for local Hugging Face embedding model.
    Loads sentence-transformers/all-MiniLM-L6-v2 once into memory.
    """
    global _embedding_model
    if _embedding_model is None:
        with _model_lock:
            if _embedding_model is None:
                settings = get_settings()
                logger.info("Initializing local SentenceTransformer model: %s", settings.embedding_model)
                try:
                    model = SentenceTransformer(settings.embedding_model)
                    _embedding_model = model
                except Exception as exc:
                    logger.error("Failed to load SentenceTransformer model '%s': %s", settings.embedding_model, exc)
                    raise RuntimeError(f"Could not initialize local embedding model: {exc}") from exc
    return _embedding_model


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """
    Generate normalized embeddings for a sequence of texts.
    Enforces exact 384 dimensions matching all-MiniLM-L6-v2.
    """
    if not texts:
        return []

    settings = get_settings()
    expected_dim = settings.embedding_dimension  # 384
    model = get_embedding_model()

    try:
        embeddings = model.encode(
            list(texts),
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    except Exception as exc:
        logger.error("Embedding generation error: %s", exc)
        raise RuntimeError(f"Embedding computation failed: {exc}") from exc

    result: list[list[float]] = []
    for idx, emb in enumerate(embeddings):
        dim = len(emb)
        if dim != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch at index {idx}: expected {expected_dim}, got {dim}"
            )
        result.append(emb.tolist())

    return result


def embed_query(query: str) -> list[float]:
    """
    Generate a normalized 384-dimensional embedding for a single search query.
    """
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Query string cannot be empty for embedding generation.")

    vectors = embed_texts([clean_query])
    return vectors[0]
