# RAG Application

This is a Retrieval Augmented Generation (RAG) application that enables users to upload documents and interact with their content through a conversational interface.

## Features

- Upload .txt and .pdf files
- Process and chunk large documents
- Conversational interface to query document content
- Support for multiple document uploads

## Running the Application

1. Ensure you have Docker and Docker Compose installed on your system.

2. Clone this repository and navigate to the project directory.

3. Create a `.env` file in the root directory and add your OpenAI API key:

   ```
   OPENAI_API_KEY=your_api_key_here
   ```

4. Build and run the Docker containers:

   ```
   docker-compose up --build
   ```

5. Open a web browser and go to `http://localhost:8501` to access the Streamlit frontend.

## Usage

1. Use the file upload interface to upload one or more .txt or .pdf files.
2. Once the files are processed, you can start asking questions in the chat interface.
3. The AI will respond with relevant information from the uploaded documents.

## Notes

- Large files may take some time to process. Please be patient during the upload and processing phase.