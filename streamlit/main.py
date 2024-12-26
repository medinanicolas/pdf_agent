import streamlit as st
import requests

# Local FastAPI server URL for file upload and document listing
UPLOAD_URL = "http://neo4j:8000/api/v1/document/upload"
LIST_DOCUMENTS_URL = "http://neo4j:8000/api/v1/document"

with st.sidebar:
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"
    
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a file to upload", type=["pdf", "docx", "txt"])

    # Handle file upload
    if uploaded_file:
        try:
            # Send the uploaded file to the API
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            response = requests.post(UPLOAD_URL, files=files)
            response.raise_for_status()

            # Display a success message
            st.sidebar.success("File uploaded successfully!")
        
        except requests.exceptions.RequestException as e:
            st.sidebar.error(f"An error occurred during the file upload: {e}")

    # List current files in the database
    st.header("Uploaded Documents")
    try:
        # Request the list of uploaded files from the API
        response = requests.get(LIST_DOCUMENTS_URL)
        response.raise_for_status()
        
        # Assuming the response is a JSON list of files
        documents = response.json().get('documents', [])  # Adjust 'documents' based on the API response structure
        if documents:
            # Display the list of uploaded files in a bullet point list with emojis
            st.markdown("Here are the documents you've uploaded:")
            for doc in documents:
                st.write(f"📄 **{doc['filename']}**")
        else:
            st.write("No files found in the database.")
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"An error occurred while fetching the file list: {e}")
