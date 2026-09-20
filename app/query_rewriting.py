import re

from app.llm import call_llm


def rewrite_query(question: str) -> str:
    """
    Rewrites a user question into one clear search query
    while preserving the original intent.
    """

    prompt = f"""Rewrite the following user question into a single, clear, well-formed
search query. Keep it faithful to the original intent - do not add new information
or assumptions. Return ONLY the rewritten query, nothing else.

Original question: "{question}"

Rewritten query:"""

    response = call_llm(prompt, max_tokens=100)

    return response["text"].strip()

def rewrite_query_with_history(
    question: str,
    history: list[dict],
) -> str:
    """
    Rewrites a user question using recent conversation history
    so follow-up questions become standalone search queries.
    """

    if not history:
        return rewrite_query(question)

    history_text = "\n".join(
        f"{message['role'].capitalize()}: {message['content']}"
        for message in history
    )

    prompt = f"""Rewrite the user's current question into a single,
clear, standalone search query using the conversation history when
necessary to resolve references such as "it", "they", "that", or
"the previous one".

Keep the original intent.
Do not add information that is not supported by the conversation.
If the question is already standalone, keep its meaning unchanged.

Return ONLY the rewritten search query, nothing else.

Conversation history:
{history_text}

Current question:
"{question}"

Rewritten query:"""

    response = call_llm(
        prompt,
        max_tokens=100,
    )

    return response["text"].strip()


def expand_query(question: str, n: int = 3) -> list[str]:
    """
    Generates multiple alternative search queries for better retrieval coverage.
    """

    prompt = f"""Generate {n} different short search queries that could help find the
answer to this question. Vary the phrasing and angle. Keep every query faithful to
the original intent. Do not add new information or assumptions.

Return ONLY a numbered list, one query per line.

Original question: "{question}"
"""

    response = call_llm(prompt, max_tokens=200)

    raw = response["text"]

    lines = []

    for line in raw.split("\n"):
        line = line.strip()

        if not line:
            continue

        line = re.sub(r"^\d+[\.\)]\s*", "", line)

        if line:
            lines.append(line)

    return lines[:n]


if __name__ == "__main__":
    question = "How pure is 24-karat gold?"

    print("\n========== QUERY REWRITING ==========")

    rewritten = rewrite_query(question)

    print("Original :", question)
    print("Rewritten:", rewritten)

    print("\n========== QUERY EXPANSION ==========")

    expanded = expand_query(question, n=3)

    for i, query in enumerate(expanded, start=1):
        print(f"{i}. {query}")