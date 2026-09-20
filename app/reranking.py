
import torch
from sentence_transformers import CrossEncoder

from app.config import CONFIG


device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nLoading reranker: {CONFIG.reranker_model}")
print(f"Reranker device: {device}")

reranker = CrossEncoder(
    CONFIG.reranker_model,
    device=device
)

print("Reranker ready.")


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = None
) -> list[dict]:
    """
    Re-scores retrieved candidates using a cross-encoder.

    The cross-encoder reads the query and candidate text together,
    producing a relevance score for each candidate.
    """

    top_k = top_k or CONFIG.rerank_k

    if not candidates:
        return []

    pairs = [
        (query, candidate["text"])
        for candidate in candidates
    ]

    rerank_scores = reranker.predict(pairs)

    for candidate, score in zip(candidates, rerank_scores):
        candidate["rerank_score"] = round(
            float(score),
            4
        )

    reranked = sorted(
        candidates,
        key=lambda candidate: candidate["rerank_score"],
        reverse=True
    )

    return reranked[:top_k]
