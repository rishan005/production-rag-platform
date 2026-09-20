import torch
from sentence_transformers import SentenceTransformer

from app.config import CONFIG

# --- Check GPU availability before loading the model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("No GPU detected - this model is small enough to run acceptably on CPU too.")

print(f"\nLoading embedding model: {CONFIG.embedding_model}")
embedder = SentenceTransformer(CONFIG.embedding_model, device=device)
print(f"Embedding dimension: {embedder.get_embedding_dimension()}")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embeds a batch of texts. Batching is faster than one-at-a-time calls."""
    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embeds a single user question."""
    return embed_texts([query])[0]