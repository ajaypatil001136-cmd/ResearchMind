
import streamlit as st
import numpy as np
from fastembed import TextEmbedding


@st.cache_resource
def get_embedding_model():
    """Load the embedding model once and reuse it."""
    return TextEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def normalize_embeddings(embeddings):
    """Normalize vectors for cosine similarity."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def generate_embeddings(chunks):
    """Convert document chunks into numerical vectors."""
    if not chunks:
        return np.empty((0, 384), dtype=np.float32)

    model = get_embedding_model()
    texts = [chunk["text"] for chunk in chunks]

    embeddings = list(model.embed(texts))

    return normalize_embeddings(embeddings)


def generate_query_embedding(question):
    """Convert a user's question into an embedding."""
    model = get_embedding_model()

    embeddings = list(model.embed([question]))

    return normalize_embeddings(embeddings)
