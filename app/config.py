import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    llm_api_key: str = os.getenv("GROQ_API_KEY", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    database_url: str = os.getenv("DATABASE_URL", "")

    llm_model: str = "openai/gpt-oss-120b"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    qdrant_url: str = os.getenv("QDRANT_URL", "")
    collection_name: str = "rag_prototype"

    chunk_size: int = 700
    chunk_overlap: int = 100

    retrieval_k: int = 20
    rerank_k: int = 5

    vector_weight: float = 0.6
    bm25_weight: float = 0.4


CONFIG = Config()