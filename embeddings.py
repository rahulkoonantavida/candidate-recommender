from sentence_transformers import SentenceTransformer
from functools import lru_cache

@lru_cache(maxsize=1)
def load_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Load and cache the SentenceTransformer model.
    """
    return SentenceTransformer(model_name)

def embed_text(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """
    Given a list of strings, return their embeddings.
    
    Args:
        texts: List of strings to encode.
        batch_size: How many documents to process at once.
    
    Returns:
        A list of embedding vectors (one per input string).
    """
    model = load_model()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    return embeddings