
import streamlit as st
from sentence_transformers import SentenceTransformer


@st.cache_resource
def get_embedding_model():
    """
    Load the embedding model once and reuse it.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(chunks):
    """
    Convert document chunks into numerical vectors.
    """

    if not chunks:
        return []

    model = get_embedding_model()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return embeddings


def generate_query_embedding(question):
    """
    Convert a user's question into an embedding.
    """

    model = get_embedding_model()

    embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return embedding
