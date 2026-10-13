import logging
import threading
from typing import Any
from pinecone import Pinecone, ServerlessSpec
from app.core.config import get_settings
from app.rag.embeddings import embed_texts, embed_query
from app.services.ingestion import DocumentChunk

logger = logging.getLogger(__name__)

_pc_client_lock = threading.Lock()
_pinecone_client: Pinecone | None = None


def get_pinecone_client() -> Pinecone:
    """
    Singleton Pinecone client initializer using PINECONE_API_KEY.
    """
    global _pinecone_client
    if _pinecone_client is None:
        with _pc_client_lock:
            if _pinecone_client is None:
                settings = get_settings()
                if not settings.pinecone_api_key:
                    raise RuntimeError("PINECONE_API_KEY is not configured in settings or environment.")
                _pinecone_client = Pinecone(api_key=settings.pinecone_api_key)
    return _pinecone_client


_verified_index: Any = None
_index_lock = threading.Lock()


def get_or_verify_index(client: Pinecone | None = None, auto_create: bool = False):
    """
    Verify existing Pinecone index configuration.
    Enforces that dimension matches local embedding model (384) and metric is cosine.
    Never deletes or drops indexes automatically.
    Reuses the verified Index object across queries to avoid redundant REST round-trips.
    """
    global _verified_index
    # If custom client passed (e.g., in unit tests), verify without caching
    if client is not None:
        return _verify_and_get_index_instance(client, auto_create=auto_create)

    if _verified_index is None:
        with _index_lock:
            if _verified_index is None:
                pc = get_pinecone_client()
                _verified_index = _verify_and_get_index_instance(pc, auto_create=auto_create)
    return _verified_index


def _verify_and_get_index_instance(pc: Pinecone, auto_create: bool = False):
    settings = get_settings()
    index_name = settings.pinecone_index_name
    desired_dim = settings.embedding_dimension  # 384

    existing_index_names = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing_index_names:
        if not auto_create:
            raise RuntimeError(
                f"Pinecone index '{index_name}' does not exist. "
                f"Please create an index with dimension={desired_dim} and metric='cosine', "
                f"or set auto_create=True during initial setup."
            )
        logger.info("Creating Pinecone index '%s' (dim=%d, metric=cosine)...", index_name, desired_dim)
        pc.create_index(
            name=index_name,
            dimension=desired_dim,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )

    # Verify existing index dimension and metric
    description = pc.describe_index(index_name)
    actual_dim = getattr(description, "dimension", None)
    if actual_dim is None and isinstance(description, dict):
        actual_dim = description.get("dimension")

    if actual_dim != desired_dim:
        raise ValueError(
            f"Pinecone index '{index_name}' has dimension {actual_dim}, but the local embedding "
            f"model requires dimension {desired_dim}. Please use an index with dimension {desired_dim}."
        )

    return pc.Index(index_name)


def upsert_chunks(
    chunks: list[DocumentChunk],
    index_client: Any = None,
    batch_size: int = 50,
) -> int:
    """
    Embed document chunks locally and upsert into Pinecone with deterministic IDs.
    Returns total number of chunks upserted.
    """
    if not chunks:
        return 0

    settings = get_settings()
    index = index_client or get_or_verify_index()
    namespace = settings.pinecone_namespace

    # Extract text from chunks and embed locally
    texts = [chunk.text for chunk in chunks]
    embeddings = embed_texts(texts)

    vectors_to_upsert: list[dict[str, Any]] = []
    for chunk, vector in zip(chunks, embeddings):
        vectors_to_upsert.append({
            "id": chunk.chunk_id,
            "values": vector,
            "metadata": chunk.metadata,
        })

    # Batch upsert to Pinecone
    total_upserted = 0
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i : i + batch_size]
        index.upsert(vectors=batch, namespace=namespace)
        total_upserted += len(batch)

    logger.info("Successfully upserted %d vector chunks to Pinecone namespace '%s'", total_upserted, namespace)
    return total_upserted


class RetrievedChunk:
    def __init__(
        self,
        text: str,
        document_name: str,
        document_id: str,
        page_number: int | None,
        chunk_index: int,
        score: float | None,
        source_type: str,
    ):
        self.text = text
        self.document_name = document_name
        self.document_id = document_id
        self.page_number = page_number
        self.chunk_index = chunk_index
        self.score = score
        self.source_type = source_type

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "document_name": self.document_name,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "score": self.score,
            "source_type": self.source_type,
        }


def retrieve_relevant_chunks(
    query: str,
    top_k: int | None = None,
    index_client: Any = None,
) -> list[RetrievedChunk]:
    """
    Embed employee question locally and query Pinecone for top_k relevant policy chunks.
    """
    settings = get_settings()
    k = top_k or settings.top_k
    index = index_client or get_or_verify_index()
    namespace = settings.pinecone_namespace

    query_vector = embed_query(query)

    query_response = index.query(
        vector=query_vector,
        top_k=k,
        namespace=namespace,
        include_metadata=True,
    )

    matches = getattr(query_response, "matches", [])
    if not matches and isinstance(query_response, dict):
        matches = query_response.get("matches", [])

    results: list[RetrievedChunk] = []
    for match in matches:
        # Match may be an object or a dict
        score = getattr(match, "score", None) if not isinstance(match, dict) else match.get("score")
        meta = getattr(match, "metadata", {}) if not isinstance(match, dict) else match.get("metadata", {})
        if not meta:
            meta = {}

        results.append(
            RetrievedChunk(
                text=meta.get("text", ""),
                document_name=meta.get("document_name", "unknown"),
                document_id=meta.get("document_id", "unknown"),
                page_number=meta.get("page_number"),
                chunk_index=meta.get("chunk_index", 0),
                score=score,
                source_type=meta.get("source_type", "document"),
            )
        )

    return results
