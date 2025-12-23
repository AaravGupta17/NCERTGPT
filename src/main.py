import os
from src.pdf_extractor import extract_text_from_pdf
from src.text_processor import clean_text, chunk_text
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline

def initialize_pipeline(data_dir: str, index_path: str = "vector_index.faiss", documents_path: str = "documents.txt") -> RAGPipeline:
    """
    Initializes the RAG pipeline by loading an existing index or creating a new one.

    Args:
        data_dir: The path to the directory containing the PDF files.
        index_path: The path to the FAISS index file.
        documents_path: The path to the documents file.

    Returns:
        An initialized RAGPipeline instance.
    """
    vector_store = VectorStore()

    if os.path.exists(index_path) and os.path.exists(documents_path):
        print("Loading existing index...")
        vector_store.load_index(index_path, documents_path)
    else:
        print("Creating new index...")
        all_text = ""
        for filename in os.listdir(data_dir):
            if filename.endswith(".pdf"):
                file_path = os.path.join(data_dir, filename)
                all_text += extract_text_from_pdf(file_path)

        if all_text:
            cleaned_text = clean_text(all_text)
            chunks = chunk_text(cleaned_text)
            vector_store.add_texts(chunks)
            print("Saving index...")
            vector_store.save_index(index_path, documents_path)
        else:
            print("No PDF files found to process.")

    return RAGPipeline(vector_store)

def main():
    """
    The main function for the chat application.
    """
    data_dir = "data"
    print("Initializing the chatbot...")
    rag_pipeline = initialize_pipeline(data_dir)
    print("Chatbot initialized. Type 'exit' to quit.")

    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            break
        answer = rag_pipeline.answer_question(query)
        print(f"Bot: {answer}")

if __name__ == "__main__":
    main()
