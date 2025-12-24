import re

def clean_text(text):
    """
    Cleans raw NCERT text by:
    - removing extra spaces
    - removing special characters
    - normalizing line breaks
    """
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\n", " ").strip()
    return text


def chunk_text(text, chunk_size=400, overlap=50):
    """
    Splits text into overlapping chunks so context is preserved.
    """

    text = clean_text(text)
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks
