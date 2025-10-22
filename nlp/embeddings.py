# nlp/embeddings.py
"""
Embeddings using sentence-transformers (all-MiniLM-L6-v2 recommended).
Returns numpy array (dtype float32) of dimension 384.
"""
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model once
_model = None


def _get_model():
    global _model
    if _model is None:
        # all-MiniLM-L6-v2 is small and fast (384 dims)
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_text(texts):
    """
    Accepts a string or list of strings. Returns numpy array or list of arrays.
    """
    model = _get_model()
    if isinstance(texts, str):
        emb = model.encode(texts, show_progress_bar=False)
        return np.array(emb, dtype=float)
    else:
        embs = model.encode(texts, show_progress_bar=False)
        return np.array(embs, dtype=float)
