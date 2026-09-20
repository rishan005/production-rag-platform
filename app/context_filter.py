from app.config import CONFIG


def filter_context(
    reranked_results: list[dict],
    min_score: float = -5.0,
) -> list[dict]:
    """
    Removes weak or irrelevant results after reranking.

    Results with a reranker score below min_score
    are excluded from the final LLM context.
    """

    filtered_results = [
        result
        for result in reranked_results
        if result["rerank_score"] >= min_score
    ]

    return filtered_results[:CONFIG.rerank_k]