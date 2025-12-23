from transformers import pipeline
from .vector_store import VectorStore

class RAGPipeline:
    """
    A Retrieval-Augmented Generation (RAG) pipeline for question answering.
    """

    def __init__(self, vector_store: VectorStore, model_name: str = "distilbert-base-cased-distilled-squad"):
        """
        Initializes the RAGPipeline.

        Args:
            vector_store: An instance of the VectorStore.
            model_name: The name of the question-answering model to use.
        """
        self.vector_store = vector_store
        self.qa_pipeline = pipeline("question-answering", model=model_name)

    def answer_question(self, query: str, k: int = 5) -> str:
        """
        Answers a question based on the documents in the vector store.

        Args:
            query: The question to answer.
            k: The number of documents to retrieve from the vector store.

        Returns:
            The generated answer.
        """
        retrieved_docs = self.vector_store.search(query, k=k)
        if not retrieved_docs:
            return "I'm sorry, I couldn't find any relevant information to answer your question."

        context = " ".join(retrieved_docs)
        result = self.qa_pipeline(question=query, context=context)
        return result['answer']
