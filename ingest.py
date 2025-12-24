import fitz  # PyMuPDF
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from utils import chunk_text

# Folder containing NCERT PDFs
PDF_FOLDER = r"C:\Users\Aarav Gupta\Downloads\NCERTGPT\NCERTGPT-feat-ncert-chatbot-8479953177274960729\data"
VECTOR_DIR = "vectorstore"
INDEX_PATH = os.path.join(VECTOR_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(VECTOR_DIR, "chunks.npy")

def extract_text_from_pdf(pdf_path):
    """Extract text from a single PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        page_text = page.get_text()
        if page_text:
            text += page_text + "\n"
    return text

def ingest():
    all_text = ""
    # Loop through all PDFs in folder
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, filename)
            print(f"📘 Processing {filename}...")
            text = extract_text_from_pdf(pdf_path)
            all_text += text + "\n"

    print("✂️ Chunking textbook content...")
    chunks = chunk_text(all_text)
    print(f"📦 Total chunks created: {len(chunks)}")

    print("🧠 Creating embeddings...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(chunks, show_progress_bar=True)

    print("💾 Building FAISS vector database...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    os.makedirs(VECTOR_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    np.save(CHUNKS_PATH, np.array(chunks, dtype=object))

    print("✅ Ingestion complete!")
    print(f"📂 Vector index saved at: {INDEX_PATH}")
    print(f"📄 Text chunks saved at: {CHUNKS_PATH}")

if __name__ == "__main__":
    ingest()
