import os
from .document_processor import DocumentProcessor

class StateManager:
    def __init__(self, upload_dir):
        self.upload_dir = upload_dir
        vector_store_path = os.path.join(os.path.dirname(upload_dir), "chroma_langchain_db")
        os.makedirs(vector_store_path, exist_ok=True)
        self.document_processor = DocumentProcessor(vector_store_path)
        self.documents = []

    def clear_session(self):
        self.clear_upload_directory()
        self.documents.clear()

    def clear_upload_directory(self):
        for filename in os.listdir(self.upload_dir):
            file_path = os.path.join(self.upload_dir, filename)
            if os.path.isfile(file_path):
                try:
                    os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")

    def process_document(self, file_path):
        result = self.document_processor.process_document(file_path)
        if result:
            self.documents.append(result['file_name'])
        return result

    def query(self, question):
        return self.document_processor.query(question)

    def get_documents(self):
        return self.documents