import os
from pydantic import BaseModel
from typing import Dict, List
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import logging


load_dotenv()

logging.basicConfig(level=logging.INFO)
base_dir = os.getcwd()
file_path = os.path.join(base_dir, "models", "all-mpnet-base-v2")
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"Model directory '{file_path}' does not exist. Please ensure the model is downloaded."
    )

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "user-input"
index = pc.Index(index_name)
model = SentenceTransformer(file_path, local_files_only=True)


def chunk_text(text: str, chunk_size: int = 200) -> List[str]:
    words = text.split()
    return [
        " ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)
    ]


class DataInput(BaseModel):
    text: str
    metadata: Dict[str, str] = {}


def store_data(data: DataInput):
    try:
        chunks = chunk_text(data.text)
        upserts = []
        hashed_id = str(hash(data.text))

        for i, chunk in enumerate(chunks):
            vector = model.encode(chunk).tolist()
            vector_id = f"{hashed_id}-{i}"
            metadata = {"text": chunk, **data.metadata}
            upserts.append((vector_id, vector, metadata))

        index.upsert(upserts)
        return {"message": "Data stored successfully", "chunks": len(upserts)}

    except Exception as e:
        logging.error(f"Error storing data: {str(e)}")


def search_similar_prompt(query: str, top_k: int = 5):
    try:
        vector = model.encode(query).tolist()
        results = index.query(vector=vector, top_k=top_k, include_metadata=True)
        return results
    except Exception as e:
        logging.error(f"Error searching for similar prompts: {str(e)}")
        return None


if __name__ == "__main__":
    sample_data = input("Type something to store data in Pinecone...")
    message = store_data(DataInput(text=sample_data, metadata={"source": "test"}))
    print(message)
