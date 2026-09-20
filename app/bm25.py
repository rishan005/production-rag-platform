import re

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class BM25Index:

    def __init__(self, chunks):
        self.chunks = chunks

        tokenized_corpus = [
            tokenize(chunk.text)
            for chunk in chunks
        ]

        self.index = BM25Okapi(tokenized_corpus)

    def search(
        self,
        query: str,
        access_level: str = "public",
    ) -> list[dict]:

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

        query_tokens = tokenize(query)

        scores = self.index.get_scores(query_tokens)

        results = []

        for chunk, score in zip(self.chunks, scores):

            # Permission check
            if chunk.access_level not in allowed_access_levels:
                continue

            results.append(
                {
                    "score": float(score),
                    "chunk_id": chunk.chunk_id,
                    "filename": chunk.filename,
                    "page": chunk.page,
                    "section": chunk.section,
                    "text": chunk.text,
                    "document_type": chunk.document_type,
                    "access_level": chunk.access_level,
                }
            )

        results.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        return results


def build_bm25_index(chunks):
    return BM25Index(chunks)