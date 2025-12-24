import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# -----------------------------
# Groq client
# -----------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------
# Absolute paths (IMPORTANT)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VECTOR_PATH = os.path.join(BASE_DIR, "vectorstore", "faiss.index")
CHUNKS_PATH = os.path.join(BASE_DIR, "vectorstore", "chunks.npy")

# -----------------------------
# Load FAISS + chunks
# -----------------------------
index = faiss.read_index(VECTOR_PATH)
chunks = np.load(CHUNKS_PATH, allow_pickle=True)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Greeting handler
# -----------------------------
def is_greeting(text):
    return text.lower().strip() in [
        "hi", "hello", "hey",
        "good morning", "good evening"
    ]


# Infer topic from history
def infer_topic_from_history(chat_history):
    for msg in reversed(chat_history):
        msg = msg.lower()
        if "solid state" in msg:
            return "solid state"
        if "electrochemistry" in msg:
            return "electrochemistry"
        if "vector" in msg:
            return "vectors"
        if "semiconductor" in msg:
            return "semiconductor"
    return None


# Main QA function
def answer_question(question, chat_history):

    # Greeting
    if is_greeting(question):
        return (
            "Hello! 👋 I am **NCERTGPT**. Ask me any question from the NCERT syllabus.",
            None
        )

    # -----------------------------
    # Conversation-aware rewriting
    # -----------------------------
    topic = infer_topic_from_history(chat_history)

    vague_phrases = [
        "explain", "explain it", "explain this",
        "like i am", "like i'm", "simplify",
        "tell me again", "give example"
    ]

    if topic and any(p in question.lower() for p in vague_phrases):
        question_for_search = f"{question} about {topic}"
    else:
        question_for_search = question

    # -----------------------------
    # FAISS retrieval
    # -----------------------------
    query_embedding = embedder.encode([question_for_search])
    _, indices = index.search(query_embedding, k=5)

    retrieved_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(retrieved_chunks)

    # Safety: weak retrieval
    if len(context.strip()) < 300:
        return (
            "I could not find a clear explanation of this topic in NCERT. "
            "Please ask a more specific NCERT syllabus question.",
            None
        )

    # -----------------------------
    # Prompt
    # -----------------------------
    system_prompt = f"""
You are NCERTGPT, an NCERT textbook tutor.

Rules:
- Answer ONLY using the NCERT content below.
- If the topic is not covered in NCERT, say:
  "This topic is not covered in NCERT."
- Adapt explanation style to the student's request
  (e.g., 'explain like I'm 5' → very simple words).
- Do NOT change the topic.
- Do NOT repeat sentences.
- Be clear and concise.

NCERT CONTENT:
{context}
"""

    messages = [{"role": "system", "content": system_prompt}]

    # Include recent conversation
    for msg in chat_history[-6:]:
        messages.append({"role": "user", "content": msg})

    messages.append({"role": "user", "content": question})

    # -----------------------------
    # Groq generation
    # -----------------------------
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.2
    )

    return response.choices[0].message.content.strip(), context
