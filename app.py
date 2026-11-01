import streamlit as st
import os
import tempfile
from rag_engine import process_document, ask_question
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.title("⚖️ Compliance Document Assistant")

# Sidebar for Ingestion
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a compliance PDF", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Chunking text and indexing in Pinecone..."):
                # Write file temporarily for the PyPDFLoader to read
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                # Process the file using our engine
                chunks_created = process_document(tmp_path)
                os.remove(tmp_path)
                st.success(f"Success! Embedded {chunks_created} chunks into Pinecone.")

# Main Chat Interface
st.header("Chat with your Documents")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask a question about the uploaded compliance policies..."):
    # Display user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching vector database..."):
            answer = ask_question(prompt)
            st.markdown(answer)
            
    st.session_state.messages.append({"role": "assistant", "content": answer})