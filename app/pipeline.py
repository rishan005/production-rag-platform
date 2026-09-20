from pathlib import Path

from app.bm25 import build_bm25_index
from app.config import CONFIG
from app.ingestion import parse_documents
from app.chunking import chunk_document
from app.vector_store import (
    qdrant,
    create_collection,
    index_chunks,
    get_collection_info,
    document_exists,
    update_document_metadata,
    create_document_id_index,
    create_metadata_indexes,
)


DOCUMENTS_PATH = Path(
    "data/documents"
)


def get_document_files() -> list[Path]:
    """
    Returns all PDF documents from the documents directory.
    """

    return sorted(
        DOCUMENTS_PATH.glob("*.pdf")
    )


def prepare_pipeline():
    """
    Loads all PDF documents, chunks them, builds BM25,
    and indexes new documents into Qdrant.
    """

    document_files = get_document_files()

    if not document_files:
        raise FileNotFoundError(
            f"No PDF documents found in {DOCUMENTS_PATH}"
        )

    print(
        f"\nFound {len(document_files)} PDF document(s)."
    )

    # -------------------------------------------------
    # 1. PARSE DOCUMENTS
    # -------------------------------------------------

    documents = parse_documents(
        [str(path) for path in document_files]
    )

    # -------------------------------------------------
    # 2. CREATE CHUNKS
    # -------------------------------------------------
    
    
    chunks = []

    for document in documents:

        if document.filename == "employee_test.pdf":
            document_type = "hr"
            access_level = "employee"
        else:
            document_type = "finance"
            access_level = "public"

        document_chunks = chunk_document(
            document,
            CONFIG,
            document_type=document_type,
            access_level=access_level,
        )

        chunks.extend(document_chunks)

    # -------------------------------------------------
    # 3. BUILD BM25 INDEX
    # -------------------------------------------------

    bm25_index = build_bm25_index(
        chunks
    )

    print(
        f"BM25 index created for "
        f"{len(chunks)} chunks"
    )

    # -------------------------------------------------
    # 4. CREATE QDRANT COLLECTION
    # -------------------------------------------------

    embedding_dim = 768

    create_collection(
        qdrant,
        CONFIG.collection_name,
        embedding_dim,
    )

    create_document_id_index(
        qdrant,
        CONFIG.collection_name,
    )

    create_metadata_indexes(
        qdrant,
        CONFIG.collection_name,
    )

    # -------------------------------------------------
    # 5. CHECK FOR DUPLICATES
    # -------------------------------------------------

    indexed_document_ids = set()

    new_document_ids = set()

    for chunk in chunks:

        document_id = chunk.document_id

        if document_id in indexed_document_ids:
            continue

        if document_exists(
            qdrant,
            CONFIG.collection_name,
            document_id,
        ):

            print(
                f"Skipping already indexed document: "
                f"{chunk.filename}"
        )

            chunk_metadata = next(
                chunk for chunk in chunks
                if chunk.document_id == document_id
            )

            update_document_metadata(
                qdrant,
                CONFIG.collection_name,
                document_id,
                document_type=chunk_metadata.document_type,
                access_level=chunk_metadata.access_level,
            )

            indexed_document_ids.add(
                document_id
        )

        else:

            new_document_ids.add(
                document_id
            )


# -------------------------------------------------
# 6. SELECT ALL CHUNKS FROM NEW DOCUMENTS
# -------------------------------------------------

    new_chunks = [
        chunk
        for chunk in chunks
        if chunk.document_id in new_document_ids
    ]
    # -------------------------------------------------
# 7. INDEX ONLY NEW DOCUMENTS
# -------------------------------------------------

    if new_chunks:

        num_indexed = index_chunks(
            qdrant,
            CONFIG.collection_name,
            new_chunks,
        )

        print(
            f"Indexed {num_indexed} new chunks "
            f"into Qdrant"
        )

    else:

        print(
            "No new documents to index."
        )


# -------------------------------------------------
# 8. COLLECTION INFORMATION
# -------------------------------------------------

    get_collection_info(
        qdrant,
        CONFIG.collection_name,
    )


    return chunks, bm25_index