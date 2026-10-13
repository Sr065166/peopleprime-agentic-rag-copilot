from unittest.mock import MagicMock
import pytest
from app.core.config import get_settings
from app.rag.vectorstore import (
    upsert_chunks,
    retrieve_relevant_chunks,
    get_or_verify_index,
)
from app.services.ingestion import DocumentChunk


def test_get_or_verify_index_success():
    settings = get_settings()
    index_name = settings.pinecone_index_name

    mock_pc = MagicMock()
    # Mock list_indexes to return configured index
    mock_idx_item = MagicMock()
    mock_idx_item.name = index_name
    mock_pc.list_indexes.return_value = [mock_idx_item]

    # Mock describe_index with dimension 384
    mock_desc = MagicMock()
    mock_desc.dimension = 384
    mock_pc.describe_index.return_value = mock_desc

    index = get_or_verify_index(client=mock_pc, auto_create=False)
    assert index is not None
    mock_pc.describe_index.assert_called_with(index_name)


def test_get_or_verify_index_dimension_mismatch():
    settings = get_settings()
    index_name = settings.pinecone_index_name

    mock_pc = MagicMock()
    mock_idx_item = MagicMock()
    mock_idx_item.name = index_name
    mock_pc.list_indexes.return_value = [mock_idx_item]

    # Return mismatched dimension (e.g. OpenAI 1536)
    mock_desc = MagicMock()
    mock_desc.dimension = 1536
    mock_pc.describe_index.return_value = mock_desc

    with pytest.raises(ValueError, match="has dimension 1536, but the local embedding model requires dimension 384"):
        get_or_verify_index(client=mock_pc, auto_create=False)


def test_get_or_verify_index_missing_no_autocreate():
    mock_pc = MagicMock()
    mock_pc.list_indexes.return_value = []

    with pytest.raises(RuntimeError, match="does not exist"):
        get_or_verify_index(client=mock_pc, auto_create=False)


def test_upsert_chunks_mocked():
    mock_index = MagicMock()
    chunks = [
        DocumentChunk(
            text="PeoplePrime leave policy summary.",
            metadata={"document_name": "leave.txt", "chunk_index": 0},
            chunk_id="leave_0",
        ),
        DocumentChunk(
            text="PeoplePrime sick leave guidelines.",
            metadata={"document_name": "leave.txt", "chunk_index": 1},
            chunk_id="leave_1",
        ),
    ]

    count = upsert_chunks(chunks, index_client=mock_index, batch_size=10)
    assert count == 2
    assert mock_index.upsert.called
    kwargs = mock_index.upsert.call_args.kwargs
    assert len(kwargs["vectors"]) == 2
    assert kwargs["vectors"][0]["id"] == "leave_0"
    assert len(kwargs["vectors"][0]["values"]) == 384
    assert kwargs["namespace"] == "peopleprime-policies"


def test_retrieve_relevant_chunks_mocked():
    mock_index = MagicMock()
    mock_match = MagicMock()
    mock_match.score = 0.92
    mock_match.metadata = {
        "text": "Employees receive 25 days of paid annual leave.",
        "document_name": "annual_leave.pdf",
        "document_id": "doc_123",
        "page_number": 2,
        "chunk_index": 3,
        "source_type": "pdf",
    }

    mock_resp = MagicMock()
    mock_resp.matches = [mock_match]
    mock_index.query.return_value = mock_resp

    results = retrieve_relevant_chunks(
        query="How many annual leave days do I get?",
        top_k=3,
        index_client=mock_index,
    )

    assert len(results) == 1
    r = results[0]
    assert r.score == 0.92
    assert r.document_name == "annual_leave.pdf"
    assert r.page_number == 2
    assert "25 days" in r.text
