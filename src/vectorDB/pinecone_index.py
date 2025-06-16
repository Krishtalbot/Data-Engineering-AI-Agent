from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)

pc_api = os.getenv("PINECONE_API_KEY")
if not pc_api:
    logging.error("PINECONE_API_KEY environment variable not set.")
    # Exit the script if the API key is not found, as it's essential.
    exit(1)

pc = Pinecone(api_key=pc_api)


def ensure_pinecone_index(
    index_name: str, dimension: int = 768, metric: str = "cosine"
):
    logging.info(f"Checking for Pinecone index: '{index_name}'...")
    if index_name not in pc.list_indexes().names():
        logging.info(f"Index '{index_name}' not found. Creating new index...")
        pc.create_index(
            name=index_name,
            dimension=dimension,  # Dimension for 'all-mpnet-base-v2' model
            metric=metric,  # Metric for similarity search
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),  # Serverless setup
        )
        logging.info(f"Pinecone index '{index_name}' created successfully.")
    else:
        logging.info(f"Pinecone index '{index_name}' already exists.")


if __name__ == "__main__":
    print("This script ensures a Pinecone index exists. You can specify a name.")
    index_to_create = input(
        "Enter the name of the Pinecone index to ensure exists (e.g., 'my-new-columns-index'): "
    )

    if index_to_create:
        ensure_pinecone_index(index_to_create)
    else:
        print("No index name provided. Exiting.")
