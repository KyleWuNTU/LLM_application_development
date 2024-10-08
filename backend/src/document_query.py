from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import logging
from typing import Optional
import os

logger = logging.getLogger(__name__)

class DocumentQuery:
    def __init__(self, vector_store, llm):
        self.vector_store = vector_store
        self.llm = llm

    def query(self, question: str, documents: list[str] = None) -> dict:
        try:
            search_kwargs = {"k": 3}
            if documents:
                # Get all documents from the vector store
                all_docs = self.vector_store.get(include=['metadatas'])
                # Filter the documents based on the filename
                filtered_sources = [
                    meta['source'] for meta in all_docs['metadatas']
                    if os.path.basename(meta['source']) in documents
                ]
                search_kwargs["filter"] = {"source": {"$in": filtered_sources}}
            
            retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)

            prompt_template = """
            You are an AI assistant that can answer questions about the provided context.
            Try to answer the question. You may reference the source documents.
            Context: {context}
            Human: {question}
            AI Assistant:"""

            prompt = ChatPromptTemplate.from_template(prompt_template)

            # Create the retrieval chain with memory
            retrieval_chain = (
                {
                    "context": retriever,
                    "question": RunnablePassthrough()
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )

            # Run the chain
            ai_message = retrieval_chain.invoke(question)

            return {
                "answer": ai_message
            }
        except Exception as e:
            logger.error(f"Error in query method: {str(e)}")
            return {
                "answer": f"An error occurred while processing your query: {str(e)}",
            }
