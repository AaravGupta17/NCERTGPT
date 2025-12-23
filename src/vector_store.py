import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class VectorStore:
    """
    A class for creating, storing, and searching vector embeddings.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the VectorStore.

        Args:
            model_name: The name of the sentence transformer model to use.
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def add_texts(self, texts: list[str]):
        """
        Adds texts to the vector store.

        Args:
            texts: A list of texts to add.
        """
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.documents.extend(texts)

    def search(self, query: str, k: int = 5) -> list[str]:
        """
        Searches for the most similar texts to a given query.

        Args:
            query: The query text.
            k: The number of results to return.

        Returns:
            A list of the most similar texts.
        """
        if self.index is None:
            return []
        query_embedding = self.model.encode([query], convert_to_tensor=False)
        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), k)
        return [self.documents[i] for i in indices[0]]

    def save_index(self, index_path: str, documents_path: str):
        """
        Saves the FAISS index and documents to disk.

        Args:
            index_path: The path to save the FAISS index.
            documents_path: The path to save the documents.
        """
        faiss.write_index(self.index, index_path)
        with open(documents_path, 'w') as f:
            for doc in self.documents:
                f.write(doc + '\n')

    def load_index(self, index_path: str, documents_path: str):
        """
        Loads a FAISS index and documents from disk.

        Args:
            index_path: The path to the FAISS index.
            documents_path: The path to the documents.
        """
        self.index = faiss.read_index(index_path)
        with open(documents_path, 'r') as f:
            self.documents = [line.strip() for line in f.readlines()]
