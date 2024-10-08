from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain.load import dumps, loads
import logging
from typing import Optional
from operator import itemgetter
import os

logger = logging.getLogger(__name__)

class DocumentQuery:
    def __init__(self, vector_store, llm):
        self.vector_store = vector_store

        self.llm = llm

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def query(self, question: str, documents: list[str] = None) -> dict:
        try:
            prompt_template = """
            You are an AI assistant that can answer questions about the provided context.
            Try to answer the question based on the context. 
            The context is a list of documents that are relevant to the question.

            Context: {context}
            Human: {question}
            AI Assistant:"""

            prompt = ChatPromptTemplate.from_template(prompt_template)

            if documents:
                # Retrieve the specified documents
                docs_contents = []
                for doc_name in documents:
                    # Retrieve documents where the 'source' metadata matches the doc_name
                    results = self.vector_store.similarity_search(
                        query="",  # Empty query since we're filtering by metadata
                        k=1,
                        filter={"file_name": doc_name}
                    )
                    if results:
                        docs_contents.append(results[0].metadata['file_name']) 
                        docs_contents.append(results[0].page_content)
                    else:
                        logger.warning(f"Document {doc_name} not found in vector store.")

                # Combine the contents of the documents
                context = "\n".join(docs_contents)

                # Log the context for inspection
                logger.info(f"Context for specific documents:\n{context}")

                # Format the prompt with the combined context
                formatted_prompt = prompt.format_messages(context=context, question=question)

                # Get the answer from the LLM
                ai_response = self.llm(formatted_prompt)

                return {
                    "answer": ai_response.content  # Extract the content from the AIMessage
                }
            else:
                search_kwargs = {"k": 3}
                retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)

                # Create the retrieval chain
                retrieval_chain = (
                    {
                        "context": retriever | self.format_docs,
                        "question": RunnablePassthrough()
                    }
                    | prompt
                    | self.llm
                    | StrOutputParser()
                )

                # Log the context for inspection
                context = retriever.get_relevant_documents(question)
                formatted_context = self.format_docs(context)
                logger.info(f"Context from retriever:\n{formatted_context}")

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
