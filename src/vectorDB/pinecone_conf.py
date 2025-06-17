import os
from pydantic import BaseModel
from typing import Dict, List
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)


class DataInput(BaseModel):
    text: str
    metadata: Dict[str, str] = {}


class PineconeManager:
    def __init__(self, index_name: str):
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        if not self.PINECONE_API_KEY:
            logging.error("PINECONE_API_KEY environment variable not set.")
            raise ValueError(
                "PINECONE_API_KEY environment variable not set. Please set it in your .env file."
            )

        self.pc = Pinecone(api_key=self.PINECONE_API_KEY)
        self.index_name = index_name

        try:
            self.index = self.pc.Index(self.index_name)
            logging.info(f"Successfully connected to Pinecone index: {self.index_name}")
        except Exception as e:
            logging.error(
                f"Error connecting to Pinecone index '{self.index_name}': {e}"
            )
            raise  # Re-raise to indicate a critical setup failure

        self._model = None
        self._load_model()
        logging.info(f"Initialized PineconeManager for index: {self.index_name}")

    def _load_model(self):
        base_dir = os.getcwd()
        model_path = os.path.join(base_dir, "models", "all-mpnet-base-v2")

        if not os.path.exists(model_path):
            logging.error(f"Model directory '{model_path}' does not exist.")
            logging.error(
                "Please ensure the 'all-mpnet-base-v2' model is downloaded and placed in a 'models' directory in your project root."
            )
            raise FileNotFoundError(
                f"SentenceTransformer model not found at '{model_path}'."
            )

        try:
            self._model = SentenceTransformer(model_path, local_files_only=True)
            logging.info(
                f"SentenceTransformer model loaded successfully from '{model_path}'."
            )
        except Exception as e:
            logging.error(f"Error loading SentenceTransformer model: {e}")
            raise

    @property
    def model(self):
        if self._model is None:
            logging.warning(
                "Model was not loaded during initialization. Attempting to load now."
            )
            self._load_model()
        return self._model

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        words = text.split()
        return [
            " ".join(words[i : i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ]

    def store_data(self, data: DataInput):
        try:
            chunks = self.chunk_text(data.text)
            upserts = []
            hashed_base_id = str(hash(f"{data.text}-{str(data.metadata)}"))

            for i, chunk in enumerate(chunks):
                # Encode the chunk using the loaded model
                vector = self.model.encode(chunk).tolist()
                # Create a unique ID for each chunk
                vector_id = f"{hashed_base_id}-{i}"
                # Combine chunk text with original metadata
                metadata = {"text": chunk, **data.metadata}
                upserts.append((vector_id, vector, metadata))

            # Perform the upsert operation to Pinecone
            self.index.upsert(vectors=upserts)  # Use 'vectors' keyword argument
            logging.info(
                f"Successfully upserted {len(upserts)} vectors to index '{self.index_name}'."
            )
            return {"message": "Data stored successfully", "chunks": len(upserts)}

        except Exception as e:
            logging.error(
                f"Error storing data in Pinecone index '{self.index_name}': {str(e)}"
            )
            raise  # Re-raise the exception for external handling

    def search_similar_prompt(self, query: str, top_k: int = 5):
        """
        Searches the configured Pinecone index for vectors similar to the query.
        """
        try:
            # Encode the query using the loaded model
            vector = self.model.encode(query).tolist()
            # Perform the query on the Pinecone index
            results = self.index.query(
                vector=vector, top_k=top_k, include_metadata=True
            )
            logging.info(
                f"Successfully queried index '{self.index_name}' for query: '{query}'"
            )
            return results
        except Exception as e:
            logging.error(
                f"Error searching for similar prompts in Pinecone index '{self.index_name}': {str(e)}"
            )
            raise  # Re-raise the exception for external handling


# Example usage for testing the PineconeManager directly (will run if this file is executed)
if __name__ == "__main__":
    test_index_name = "test-user-input"  # Use a distinct name for direct testing

    # Ensure the test index exists before proceeding (this would typically be handled by pinecone_index.py)
    pc_api = os.getenv("PINECONE_API_KEY")
    pc_client = Pinecone(api_key=pc_api)
    if test_index_name not in pc_client.list_indexes().names():
        print(f"Creating index '{test_index_name}' for testing...")
        pc_client.create_index(
            name=test_index_name,
            dimension=768,  # Ensure this matches your model's embedding dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Index '{test_index_name}' created.")
    else:
        print(f"Index '{test_index_name}' already exists.")

    try:
        # Initialize the manager with the test index name
        manager = PineconeManager(test_index_name)

        sample_data = input(
            "Type something to store data in the TEST Pinecone index (or 'exit'): "
        )
        if sample_data.lower() != "exit":
            message = manager.store_data(
                DataInput(text=sample_data, metadata={"source": "pinecone_conf_test"})
            )
            print(message)

            query = input(
                "Type a query to search the TEST Pinecone index (or 'exit'): "
            )
            if query.lower() != "exit":
                results = manager.search_similar_prompt(query, top_k=3)
                if results and results.matches:
                    print("\nTest Search Results:")
                    for match in results.matches:
                        print(
                            f"  Score: {match.score:.4f}, Text: {match.metadata.get('text', 'N/A')}"
                        )
                else:
                    print("No test search results found.")
    except Exception as e:
        print(f"An error occurred during PineconeManager test: {e}")
