import streamlit as st
from qa import answer_question

# Page config
st.set_page_config(
    page_title="NCERTGPT",
    page_icon="📘",
    layout="wide"
)

# Title
st.title("📘 NCERTGPT")
st.caption("A conversational AI tutor trained strictly on NCERT textbooks")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input box (chat-style)
user_input = st.chat_input("Ask a question from the NCERT syllabus...")

if user_input:
    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    st.session_state.chat_history.append(f"Student: {user_input}")

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking from NCERT 📖..."):
            answer, source = answer_question(
                user_input,
                st.session_state.chat_history
            )

            st.markdown(answer)

            # Show NCERT source if available
            if source:
                with st.expander("📚 NCERT source used"):
                    st.write(source)

    # Store assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
    st.session_state.chat_history.append(f"NCERTGPT: {answer}")
