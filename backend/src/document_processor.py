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
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self, persist_directory):
        # Set up embeddings
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
        )
        self.llm = ChatOpenAI(model_name="gpt-4o")

    def process_document(self, file_path):
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


            # Add metadata to chunks
            for i, chunk in enumerate(chunks):
                chunk.metadata['source'] = file_path
                chunk.metadata['chunk_id'] = i

            # Add documents to the vector store
            self.vector_store.add_documents(documents=chunks)
            logger.info(f"Vector store size after adding documents: {len(self.vector_store.get())}")


            logger.info(f"Document processed: {len(chunks)} chunks created")
            return {
                "file_path": file_path,
                "file_name": os.path.basename(file_path), 
                "num_chunks": len(chunks),
                "vector_store_size": len(self.vector_store.get()),
                "num_documents": len(documents),
            }
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return None

    def query(self, question: str) -> dict:
        try:
            retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            prompt_template = """
            You are an AI assistant that can answer questions about the provided context.
            Try to answer the question based on the provided context. Include references to the source documents.
            Context: {context}
            Question: {question}
            Answer:"""
            prompt = ChatPromptTemplate.from_template(prompt_template)

            # Create the retrieval chain
            retrieval_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )

            # Run the chain
            logger.info("Invoking retrieval chain")
            answer = retrieval_chain.invoke(question)
            logger.info(f"Retrieval chain response: {answer}")
            
            logger.info(f"Processed answer: {answer}")
            
            return {
                "answer": answer
            }
        except Exception as e:
            logger.error(f"Error in query method: {str(e)}")
            return {
                "answer": f"An error occurred while processing your query: {str(e)}"
            }