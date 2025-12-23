import re

def clean_text(text: str) -> str:
    """
    Cleans the input text by removing extra whitespace and special characters.

    Args:
        text: The text to clean.

    Returns:
        The cleaned text.
    """
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_text(text: str, chunk_size: int = 1024, overlap: int = 128) -> list[str]:
    """
    Splits the text into smaller chunks with a specified overlap.

    Args:
        text: The text to chunk.
        chunk_size: The desired size of each chunk.
        overlap: The number of characters to overlap between chunks.

    Returns:
        A list of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
