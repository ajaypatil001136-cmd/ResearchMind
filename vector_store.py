
import faiss
import numpy as np


# ==================================================
# 1. CREATE FAISS INDEX
# ==================================================

def create_index(embeddings):
    """
    Create a FAISS index using cosine similarity.

    Cosine similarity is implemented using:
    1. L2 normalization
    2. Inner product search
    """

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    # Validate embeddings
    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a 2D array."
        )

    if embeddings.shape[0] == 0:
        raise ValueError(
            "Cannot create an index with empty embeddings."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embeddings contain invalid numerical values."
        )

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)

    # Get embedding dimensions
    dimension = embeddings.shape[1]

    # Create FAISS inner product index
    index = faiss.IndexFlatIP(dimension)

    # Add vectors to index
    index.add(embeddings)

    return index


# ==================================================
# 2. SEARCH FAISS INDEX
# ==================================================

def search_index(index, chunks, query_embedding, top_k=3):
    """
    Search for the most relevant document chunks.

    Uses cosine similarity through normalized vectors.
    Returns:
    - Text
    - Page number
    - Source filename
    - Similarity score
    """

    # Validate index and chunks
    if index is None:
        raise ValueError(
            "FAISS index is not initialized."
        )

    if not chunks:
        return []

    if index.ntotal != len(chunks):
        raise ValueError(
            "FAISS index size does not match chunk count."
        )

    # Convert query embedding to float32
    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    # Ensure correct dimensions
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    if query_embedding.ndim != 2:
        raise ValueError(
            "Query embedding must be a 1D or 2D array."
        )

    if query_embedding.shape[1] != index.d:
        raise ValueError(
            f"Embedding dimension mismatch. "
            f"Expected {index.d}, "
            f"received {query_embedding.shape[1]}."
        )

    if not np.isfinite(query_embedding).all():
        raise ValueError(
            "Query embedding contains invalid values."
        )

    # Normalize query for cosine similarity
    faiss.normalize_L2(query_embedding)

    # Determine number of results
    k = min(top_k, index.ntotal)

    if k <= 0:
        return []

    # Search FAISS
    scores, indices = index.search(
        query_embedding,
        k
    )

    results = []

    # Process retrieved chunks
    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:
            continue

        chunk = chunks[int(idx)]

        results.append({

            "text": chunk["text"],

            "page": chunk["page"],

            "source": chunk.get(
                "source",
                "Unknown PDF"
            ),

            "score": float(score)

        })

    return results
