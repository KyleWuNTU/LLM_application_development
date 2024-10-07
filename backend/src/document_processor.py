from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
import os
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self, vector_store: Chroma, llm):
        self.vector_store = vector_store
        self.llm = llm

    def process_document(self, file_path: str) -> dict:
        logger.info(f"Processing document: {file_path}")
        
        try:
            # Load the document
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension == '.txt':
                loader = TextLoader(file_path)
            elif file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            documents = loader.load()
            if not documents:
                logger.error("No documents were loaded. File parsing may have failed.")
                return None

            # Split the documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Chunk example: {chunks[0].page_content if chunks else 'No chunks'}")

            timestamp = datetime.now().isoformat()
            # Add metadata to chunks
            for i, chunk in enumerate(chunks):
                chunk.metadata['source'] = file_path
                chunk.metadata['chunk_id'] = i
                chunk.metadata['timestamp'] = timestamp
            # Add documents to the vector store
            self.vector_store.add_documents(documents=chunks)
            logger.info(f"Vector store size after adding documents: {len(self.vector_store.get())}")
            
            return {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "num_chunks": len(chunks),
                "vector_store_size": len(self.vector_store.get()),
            }
        
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return None