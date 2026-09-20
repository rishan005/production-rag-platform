from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

from app.config import CONFIG
from app.chunking import Chunk
from app.embeddings import embed_texts


# Connect to Qdrant Cloud
qdrant = QdrantClient(
    url=CONFIG.qdrant_url,
    api_key=CONFIG.qdrant_api_key,
)


def create_collection(
    client: QdrantClient,
    name: str,
    vector_size: int
):
    """Creates the Qdrant collection if it does not already exist."""

    collections = client.get_collections().collections

    if name not in [collection.name for collection in collections]:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

        print(f"Created collection: {name}")

    else:
        print(f"Collection already exists: {name}")


def _make_point_id(chunk_id: str) -> str:
    """
    Creates a deterministic UUID from the chunk ID.

    The same chunk will always receive the same Qdrant point ID.
    This prevents duplicate points when ingestion is run again.
    """
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            chunk_id
        )
    )


def index_chunks(
    client: QdrantClient,
    collection: str,
    chunks: list[Chunk],
):
    """Embeds chunks and inserts/updates them in Qdrant."""

    texts = [c.text for c in chunks]

    vectors = embed_texts(texts)

    points = [
        PointStruct(
            id=_make_point_id(c.chunk_id),
            vector=vector,
            payload={
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "filename": c.filename,
                "page": c.page,
                "section": c.section,
                "text": c.text,

                "document_type": c.document_type,
                "access_level": c.access_level,

            },
        )
        for c, vector in zip(chunks, vectors)
    ]

    client.upsert(
        collection_name=collection,
        points=points,
    )

    return len(points)


def get_collection_info(
    client: QdrantClient,
    collection: str
):
    """Returns information about the Qdrant collection."""

    info = client.get_collection(collection)

    print(
        f"Collection '{collection}' contains "
        f"{info.points_count} points"
    )

    print(
        f"Vector size: "
        f"{info.config.params.vectors.size}"
    )

    return info

def document_exists(
    qdrant,
    collection_name: str,
    document_id: str,
) -> bool:
    """
    Checks whether a document has already been indexed
    in the Qdrant collection.
    """

    response = qdrant.scroll(
        collection_name=collection_name,
        scroll_filter={
            "must": [
                {
                    "key": "document_id",
                    "match": {
                        "value": document_id
                    }
                }
            ]
        },
        limit=1,
        with_payload=False,
        with_vectors=False,
    )

    points, _ = response

    return len(points) > 0

def update_document_metadata(
    qdrant,
    collection_name: str,
    document_id: str,
    document_type: str,
    access_level: str,
):
    """
    Updates metadata for all chunks belonging to a document.

    This does not re-embed or create new points.
    """

    response = qdrant.scroll(
        collection_name=collection_name,
        scroll_filter={
            "must": [
                {
                    "key": "document_id",
                    "match": {
                        "value": document_id
                    },
                }
            ]
        },
        limit=100,
        with_payload=False,
        with_vectors=False,
    )

    points, _ = response

    point_ids = [
        point.id
        for point in points
    ]

    if not point_ids:
        print(
            f"No points found for document: "
            f"{document_id}"
        )
        return

    qdrant.set_payload(
        collection_name=collection_name,
        payload={
            "document_type": document_type,
            "access_level": access_level,
        },
        points=point_ids,
    )

    print(
        f"Updated metadata for document: "
        f"{document_id}"
    )

def create_document_id_index(
    qdrant,
    collection_name: str,
):
    """
    Creates a keyword payload index on document_id
    if it does not already exist.
    """

    from qdrant_client.models import PayloadSchemaType

    try:

        qdrant.create_payload_index(
            collection_name=collection_name,
            field_name="document_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )

        print(
            "Created payload index for document_id."
        )

    except Exception as e:

        if "already exists" in str(e).lower():

            print(
                "Payload index for document_id already exists."
            )

        else:

            raise

def create_metadata_indexes(
    qdrant,
    collection_name: str,
):
    """
    Creates payload indexes required for
    metadata filtering and document permissions.
    """

    from qdrant_client.models import PayloadSchemaType

    fields = [
        "document_type",
        "access_level",
    ]

    for field in fields:

        try:

            qdrant.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )

            print(
                f"Created payload index for {field}."
            )

        except Exception as e:

            if "already exists" in str(e).lower():

                print(
                    f"Payload index for {field} "
                    f"already exists."
                )

            else:

                raise