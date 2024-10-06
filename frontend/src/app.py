import streamlit as st
import requests
import os
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BACKEND_URL = os.getenv("BACKEND_URL")

st.title("RAG Application")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize document list
if "documents" not in st.session_state:
    st.session_state.documents = []

# Document selection
st.sidebar.header("Document Selection")
selected_documents = st.sidebar.multiselect(
    "Select documents to search",
    options=st.session_state.documents,
    default=st.session_state.documents
)       

# File upload section
uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf"], accept_multiple_files=True)

if uploaded_file is not None:
    for file in uploaded_file:
        with st.spinner(f"Uploading and processing {file.name}..."):
            files = {"file": (file.name, file.getvalue(), file.type)}
            response = requests.post(f"{BACKEND_URL}/upload", files=files)
            if response.status_code == 200:
                result = response.json()
                st.success(f"File {result['filename']} uploaded and processed successfully!")
                st.info(f"Number of chunks: {result['num_chunks']}")
                st.info(f"Vector store size: {result['vector_store_size']}")
                # Add the file name to the documents list if it's not already there
                if result['filename'] not in st.session_state.documents:
                    st.session_state.documents.append(result['filename'])
            else:
                st.error(f"Error uploading file {file.name}")

# Chat interface
st.header("Chat with your documents")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            st.info(f"Sources: {', '.join(message['sources'])}")

# React to user input
if prompt := st.chat_input("What would you like to know?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Send user's question to the backend
    response = requests.post(f"{BACKEND_URL}/query", json={"question": prompt, "documents": selected_documents})

    if response.status_code == 200:
        result = response.json()
        logger.info(f"API Response: {result}")
        
        answer_data = result.get("response", {})
        if not isinstance(answer_data, dict):
            logger.error(f"Unexpected answer_data type: {type(answer_data)}")
            answer_data = {}
        
        answer = answer_data.get("answer", "Sorry, I couldn't find an answer.")
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(answer)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        # User feedback
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Helpful"):
                st.success("Thank you for your feedback!")
        with col2:
            if st.button("👎 Not Helpful"):
                st.error("We'll try to improve. Thank you for your feedback!")
    else:
        st.error(f"Error querying the document: {response.text}")