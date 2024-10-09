# RAG Application

This is a Retrieval Augmented Generation (RAG) application that enables users to upload documents and interact with their content through a conversational interface.

## Features

- Upload and process .txt and .pdf files
- Chunking of large documents for efficient processing
- Conversational interface to query document content
- Support for multiple document uploads
- Real-time document list refresh
- User feedback mechanism for responses (thumbs up/down)
- Memory mechanism for maintaining context in conversations

## Prerequisites

- Docker
- Docker Compose
- Git

## Setup and Running the Application

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a `.env` file in the root directory with the following content:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   BACKEND_PORT=8000
   FRONTEND_PORT=8501
   BACKEND_URL=http://backend:8000
   ```
   Replace `your_openai_api_key_here` with your actual OpenAI API key.

3. Build and run the Docker containers:
   ```
   docker-compose up --build
   ```

4. Once the containers are up and running, open a web browser and navigate to:
   ```
   http://localhost:8501
   ```
   This will open the Streamlit frontend of the application.

## Using the Devcontainer Setup

If you're using Visual Studio Code with the Remote - Containers extension:

1. Open the project folder in VS Code.
2. When prompted, click "Reopen in Container" or use the command palette (F1) and select "Remote-Containers: Reopen in Container".
3. VS Code will build the devcontainer and open the project inside it.

The devcontainer is configured with all necessary extensions and dependencies.

## Application Structure

The application consists of two main components:

1. Backend (FastAPI):
   - Handles document processing and storage
   - Manages the vector store for efficient document retrieval
   - Processes queries and generates responses
   - Implements a memory mechanism to maintain conversation context

2. Frontend (Streamlit):
   - Provides a user-friendly interface for document upload and interaction
   - Displays chat history and responses
   - Allows document selection for targeted queries

## Usage

1. Upload Documents:
   - Use the file uploader to add .txt or .pdf files to the system.
   - The application will process and chunk the documents automatically.

2. Query Documents:
   - Type your questions in the chat input at the bottom of the page.
   - The AI will respond with relevant information from the uploaded documents.
   - The memory mechanism allows the AI to maintain context across multiple queries, providing more coherent and contextually relevant responses.

3. Document Selection:
   - Use the sidebar to select specific documents for querying.
   - Click the "Refresh Document List" button to update the available documents.

4. Feedback:
   - After each AI response, you can provide feedback using the thumbs up/down buttons.

## Notes

- Large files may take some time to process. Please be patient during the upload and processing phase.
- The application uses OpenAI's GPT-4 model for generating responses, ensure your API key has the necessary permissions.

## Troubleshooting

If you encounter any issues:
1. Ensure all environment variables are correctly set in the `.env` file.
2. Check Docker logs for any error messages:
   ```
   docker-compose logs
   ```
3. Verify that both frontend and backend containers are running:
   ```
   docker-compose ps
   ```

For more detailed information about the implementation, refer to the source code in the `backend/src` and `frontend/src` directories.