import os
from sentence_transformers import SentenceTransformer


def ensure_local_model(model_name="all-mpnet-base-v2", base_dir="models"):
    """
    Checks if the SentenceTransformer model is saved locally.
    If not, downloads and saves it to the specified directory.
    """
    local_path = os.path.join(base_dir, model_name)

    # Check if the model directory exists
    if os.path.exists(local_path):
        print(f" Model '{model_name}' already exists at '{local_path}'")
    else:
        print(f"Model '{model_name}' not found locally. Downloading and saving...")
        model = SentenceTransformer(f"sentence-transformers/{model_name}")
        os.makedirs(base_dir, exist_ok=True)
        model.save(local_path)
        print(f" Model saved to '{local_path}'")

    return local_path


if __name__ == "__main__":
    ensure_local_model()
