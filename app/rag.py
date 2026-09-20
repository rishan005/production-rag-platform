from app.config import CONFIG
from app.retrival import multi_query_hybrid_search
from app.reranking import rerank
from app.context_filter import filter_context
from app.generation import generate_answer
from app.memory import ConversationMemory
from app.query_rewriting import rewrite_query_with_history

class RAGPipeline:
    """
    Central orchestration layer for the Production RAG system.

    Pipeline:

    Query
      ↓
    Multi-query hybrid retrieval
      ↓
    Cross-encoder reranking
      ↓
    Context filtering
      ↓
    LLM generation
      ↓
    Citation verification
    """

    def __init__(self, chunks, bm25_index):
        self.chunks = chunks
        self.bm25_index = bm25_index
        self.memory = ConversationMemory(max_messages=10)

    def answer(
            self,
            query: str,
            access_level: str = "public",
            ) -> dict:

        # -------------------------------------------------
        # 1. QUERY REWRITING WITH CONVERSATION MEMORY
        # -------------------------------------------------

        history = self.memory.get_history()

        search_query = rewrite_query_with_history(
            question=query,
            history=history,
        )


        # -------------------------------------------------
        # 1. HYBRID RETRIEVAL
        # -------------------------------------------------

        hybrid_results = multi_query_hybrid_search(
            query=search_query,
            chunks=self.chunks,
            bm25_index=self.bm25_index,
            top_k=CONFIG.retrieval_k,
            access_level=access_level,
            )

        # -------------------------------------------------
        # 2. RERANKING
        # -------------------------------------------------

        reranked_results = rerank(
            query=search_query,
            candidates=hybrid_results,
            top_k=CONFIG.rerank_k,
        )

        # -------------------------------------------------
        # 3. CONTEXT FILTERING
        # -------------------------------------------------

        filtered_results = filter_context(
            reranked_results
        )

        # -------------------------------------------------
        # 4. ANSWER GENERATION
        # -------------------------------------------------

        result = generate_answer(
            query=query,
            reranked_results=filtered_results,
        )

        # -------------------------------------------------
        # 5. ADD PIPELINE INFORMATION
        # -------------------------------------------------

        result["query"] = query

        result["retrieval"] = {
            "method": "multi_query_hybrid",
            "retrieval_k": CONFIG.retrieval_k,
            "rerank_k": CONFIG.rerank_k,
        }


# -------------------------------------------------
# 6. SAVE CONVERSATION MEMORY
# -------------------------------------------------

        self.memory.add_user_message(query)
        self.memory.add_assistant_message(result["answer"])

        return result