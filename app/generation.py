from app.config import CONFIG
from app.llm import call_llm
from app.citations import verify_citations


SYSTEM_PROMPT = """
You are a helpful RAG assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Do not make up information.
- If the answer cannot be found in the context, say:
  "I don't have enough information in the provided documents."
- Keep the answer clear and concise.
- Use the context as the source of truth.
"""


def build_context(results: list[dict]) -> str:
    """
    Converts reranked search results into a context
    that can be provided to the LLM.
    """

    context_parts = []

    for i, result in enumerate(results, start=1):
        context_parts.append(
            f"""
--- Context {i} ---
Document: {result['filename']}
Page: {result['page']}
Section: {result['section']}

{result['text']}
"""
        )

    return "\n".join(context_parts)


def generate_answer(query: str, reranked_results: list[dict]) -> dict:

    # --------------------------------------------------
    # HANDLE EMPTY CONTEXT
    # --------------------------------------------------

    if not reranked_results:
        return {
            "answer": "I don't have enough information in the provided documents.",
            "input_tokens": 0,
            "output_tokens": 0,
            "sources": [],
            "citation_verification": {
                "supported": True,
                "source_ids": [],
                "reason": "No accessible sources were available for this query.",
            },
        }

    # --------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------

    context = build_context(reranked_results)

    prompt = f"""
Context:
{context}

Question:
{query}

Answer:
"""

    # --------------------------------------------------
    # CALL LLM
    # --------------------------------------------------

    response = call_llm(
        prompt=prompt,
        max_tokens=CONFIG.rerank_k * 100,
        system=SYSTEM_PROMPT,
    )

    answer = response["text"]

    # --------------------------------------------------
    # BUILD SOURCES
    # --------------------------------------------------

    sources = [
        {
            "filename": result["filename"],
            "page": result["page"],
            "section": result["section"],
            "chunk_id": result["chunk_id"],
            "text": result["text"],
        }
        for result in reranked_results
    ]

    # --------------------------------------------------
    # VERIFY CITATIONS
    # --------------------------------------------------

    citation_verification = verify_citations(
        answer=answer,
        sources=sources,
    )

    # --------------------------------------------------
    # RETURN RESULT
    # --------------------------------------------------

    return {
        "answer": answer,
        "input_tokens": response["input_tokens"],
        "output_tokens": response["output_tokens"],
        "sources": [
            {
                "filename": source["filename"],
                "page": source["page"],
                "section": source["section"],
                "chunk_id": source["chunk_id"],
            }
            for source in sources
        ],
        "citation_verification": citation_verification,
    }