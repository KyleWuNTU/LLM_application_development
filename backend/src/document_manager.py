import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from .document_processor import DocumentProcessor
from .document_query import DocumentQuery
from langchain_openai import ChatOpenAI
import logging

logger = logging.getLogger(__name__)

class DocumentManager:
    def __init__(self, upload_dir: str):
        self.upload_dir = upload_dir
        backend_dir = os.path.dirname(os.path.dirname(__file__))  # Get the backend directory
        vector_store_path = os.path.join(backend_dir, "chroma_langchain_db")
        os.makedirs(vector_store_path, exist_ok=True)
        
        # Initialize the embeddings, vector store, and llm
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_store = Chroma(collection_name="rag_collection", persist_directory=vector_store_path, embedding_function=self.embeddings)
        self.llm = ChatOpenAI(model_name="gpt-4o")
        
        # Initialize document processor and query with the same vector store
        self.document_processor = DocumentProcessor(self.vector_store, self.llm)
        self.document_query = DocumentQuery(self.vector_store, self.llm)

    def _load_existing_documents(self) -> list[str]:
        # Retrieve only the metadatas from the vector store
        results = self.vector_store.get(include=['metadatas'])
        logger.info(f"Retrieved metadata for {len(results['metadatas'])} documents from vector store")
        
        if not results['metadatas']:
            logger.info("No documents found in vector store")
            return []
        
        # Extract unique filenames from the metadata
        filenames = set()
        for metadata in results['metadatas']:
            source = metadata.get('source')
            if source:
                filename = os.path.basename(source)
                filenames.add(filename)
            else:
                logger.warning(f"Document metadata missing 'source': {metadata}")
        
        unique_filenames = list(filenames)
        logger.info(f"Unique filenames: {unique_filenames}")
        
        return unique_filenames

    def process_document(self, file_path: str) -> dict:
        result = self.document_processor.process_document(file_path)
        if result:
            file_name = result['file_name']
            documents = self._load_existing_documents()
            if file_name not in documents:
                documents.append(file_name)
            logger.info(f"Processed {file_name}. Total unique documents: {len(documents)}")
        return result

    def query(self, question: str, documents: list[str] = None) -> dict:
        answer = self.document_query.query(question, documents)
        return answer

    def get_all_documents(self) -> list[str]:
        return self._load_existing_documents()
