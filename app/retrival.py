from app.config import CONFIG
from app.embeddings import embed_query
from app.vector_store import qdrant
from app.query_rewriting import rewrite_query, expand_query


def semantic_search(
    query: str,
    top_k: int = None,
    access_level: str = "public",
) -> list[dict]:
    """
    Retrieves semantically similar chunks from Qdrant
    while respecting document access permissions.
    """

    top_k = top_k or CONFIG.retrieval_k

    query_vector = embed_query(query)

    # Define which document access levels the user can see
    allowed_levels = {
        "public": ["public"],
        "employee": ["public", "employee"],
        "admin": ["public", "employee", "admin"],
    }

    if access_level not in allowed_levels:
        raise ValueError(
            f"Invalid access level: {access_level}"
        )

    response = qdrant.query_points(
        collection_name=CONFIG.collection_name,
        query=query_vector,
        query_filter={
            "should": [
                {
                    "key": "access_level",
                    "match": {
                        "value": level
                    },
                }
                for level in allowed_levels[access_level]
            ]
        },
        limit=top_k,
    )

    return [
        {
            "score": round(r.score, 4),
            "chunk_id": r.payload["chunk_id"],
            "filename": r.payload["filename"],
            "page": r.payload["page"],
            "section": r.payload["section"],
            "text": r.payload["text"],
            "document_type": r.payload.get(
                "document_type",
                "general"
            ),
            "access_level": r.payload.get(
                "access_level",
                "public"
            ),
        }
        for r in response.points
    ]

def truncate_at_word(text: str, max_chars: int = 200) -> str:
    """Truncates text without cutting a word in half."""

    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")

    if last_space == -1:
        return truncated

    return truncated[:last_space]


def _min_max_normalize(scores: list[float]) -> list[float]:
    """Rescales scores to the 0-1 range."""

    if not scores:
        return scores

    lo = min(scores)
    hi = max(scores)

    if hi - lo < 1e-9:
        return [1.0 for _ in scores]

    return [
        (score - lo) / (hi - lo)
        for score in scores
    ]


def hybrid_search(
    query: str,
    chunks,
    bm25_index,
    top_k: int = None,
    vector_weight: float = None,
    bm25_weight: float = None,
    access_level: str = "public",
) -> list[dict]:
    """
    Combines semantic similarity from Qdrant
    with BM25 keyword matching.
    """

    top_k = top_k or CONFIG.retrieval_k

    vector_weight = (
        vector_weight
        if vector_weight is not None
        else CONFIG.vector_weight
    )

    bm25_weight = (
        bm25_weight
        if bm25_weight is not None
        else CONFIG.bm25_weight
    )

    # --------------------------------------------------
    # 1. SEMANTIC SEARCH
    # --------------------------------------------------

    query_vector = embed_query(query)

    # Define which document access levels the user can see
    allowed_levels = {
        "public": ["public"],
        "employee": ["public", "employee"],
        "admin": ["public", "employee", "admin"],
    }

    if access_level not in allowed_levels:
        raise ValueError(
            f"Invalid access level: {access_level}"
        )

    semantic_response = qdrant.query_points(
        collection_name=CONFIG.collection_name,
        query=query_vector,
        query_filter={
            "should": [
                {
                    "key": "access_level",
                    "match": {
                        "value": level
                    },
                }
                for level in allowed_levels[access_level]
            ]
        },
        limit=len(chunks),
    )

    semantic_scores_by_chunk_id = {
        r.payload["chunk_id"]: r.score
        for r in semantic_response.points
    }

    semantic_scores = [
        semantic_scores_by_chunk_id.get(
            chunk.chunk_id,
            0.0
        )
        for chunk in chunks
    ]

    # --------------------------------------------------
    # 2. BM25 SEARCH
    # --------------------------------------------------

    bm25_results = bm25_index.search(
        query=query,
        access_level=access_level,
    )

    bm25_scores_by_chunk_id = {
        result["chunk_id"]: result["score"]
        for result in bm25_results
    }

    bm25_scores = [
        bm25_scores_by_chunk_id.get(chunk.chunk_id, 0.0)
        for chunk in chunks
    ]

    # --------------------------------------------------
    # 3. NORMALIZE SCORES
    # --------------------------------------------------

    norm_semantic = _min_max_normalize(
        semantic_scores
    )

    norm_bm25 = _min_max_normalize(
        bm25_scores
    )

    # --------------------------------------------------
    # 4. COMBINE SCORES
    # --------------------------------------------------

    combined = []

    allowed_levels = {
        "public": ["public"],
        "employee": ["public", "employee"],
        "admin": ["public", "employee", "admin"],
    }

    if access_level not in allowed_levels:
        raise ValueError(
            f"Invalid access level: {access_level}"
        )

    allowed_access_levels = allowed_levels[access_level]

    for i, chunk in enumerate(chunks):

    # Skip documents the current user is not allowed to access
        if chunk.access_level not in allowed_access_levels:
            continue

        final_score = (
            vector_weight * norm_semantic[i]
            + bm25_weight * norm_bm25[i]
        )

        combined.append(
            {
                "score": round(final_score, 4),
                "semantic_score": round(
                    semantic_scores[i], 4
                ),
                "bm25_score": round(
                    bm25_scores[i], 4
                ),
                "chunk_id": chunk.chunk_id,
                "filename": chunk.filename,
                "page": chunk.page,
                "section": chunk.section,
                "text": chunk.text,
                "document_type": chunk.document_type,
                "access_level": chunk.access_level,
            }
        )

    # --------------------------------------------------
    # 5. SORT AND RETURN TOP-K
    # --------------------------------------------------

    combined.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return combined[:top_k]

# ==================================================
# MULTI-QUERY HYBRID SEARCH
# ==================================================

def multi_query_hybrid_search(
    query: str,
    chunks,
    bm25_index,
    top_k: int = None,
    access_level: str = "public",
) -> list[dict]:

    top_k = top_k or CONFIG.retrieval_k

    # 1. Query rewriting
    rewritten_query = rewrite_query(query)

    # 2. Query expansion
    expanded_queries = expand_query(query, n=3)

    # 3. Build search query list
    search_queries = [
        query,
        rewritten_query,
        *expanded_queries,
    ]

    # Remove duplicate queries
    search_queries = list(dict.fromkeys(search_queries))

    # 4. Run hybrid search for every query
    all_results = {}

    for search_query in search_queries:

        results = hybrid_search(
            query=search_query,
            chunks=chunks,
            bm25_index=bm25_index,
            top_k=top_k,
            access_level=access_level,
            )

        for result in results:

            chunk_id = result["chunk_id"]

            if chunk_id not in all_results:
                all_results[chunk_id] = result

            elif result["score"] > all_results[chunk_id]["score"]:
                all_results[chunk_id] = result

    # 5. Sort final results
    results = sorted(
        all_results.values(),
        key=lambda x: x["score"],
        reverse=True,
    )

    # 6. Return top-K
    return results[:top_k]