import json

from app.llm import call_llm

def verify_citations(
    answer: str,
    sources: list[dict],
) -> dict:
    """
    Verifies whether the generated answer is supported
    by the retrieved source chunks.
    """

    source_context = "\n\n".join(
        f"""
Source {i}:
Filename: {source['filename']}
Page: {source['page']}
Section: {source['section']}
Chunk ID: {source['chunk_id']}

Text:
{source['text']}
"""
        for i, source in enumerate(sources, start=1)
    )

    prompt = f"""
You are a citation verification system for a RAG application.

Generated Answer:
{answer}

Retrieved Sources:
{source_context}

Determine whether the generated answer is fully supported
by the retrieved sources.

Return ONLY valid JSON:

{{
    "supported": true,
    "source_ids": ["chunk_id"],
    "reason": "short explanation"
}}

Rules:

- supported = true only if the answer is supported by
  the retrieved source text.
- supported = false if the answer contains unsupported
  factual information.
- source_ids must contain the chunk IDs that support the answer.
- Do not use outside knowledge.
- Do not add markdown.
- Do not add text outside the JSON.
"""

    response = call_llm(
        prompt=prompt,
        max_tokens=250,
        system="You are a strict citation verification system.",
    )

    raw_result = response["text"].strip()

    try:
        return json.loads(raw_result)

    except json.JSONDecodeError:

        print("Warning: Citation verifier returned invalid JSON:")
        print(raw_result)

        return {
            "supported": False,
            "source_ids": [],
            "reason": "Citation verifier returned invalid JSON.",
        }