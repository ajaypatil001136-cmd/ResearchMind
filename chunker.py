
def create_chunks(text, page_number, chunk_size=500, overlap=100):
    """
    Split page text into smaller overlapping chunks.
    Each chunk retains its original PDF page number.
    """

    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk_text = " ".join(words[start:end])

        chunks.append({
            "text": chunk_text,
            "page": page_number
        })

        if end >= len(words):
            break

        start = end - overlap

    return chunks
