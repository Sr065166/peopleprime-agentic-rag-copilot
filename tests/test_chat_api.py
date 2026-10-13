from unittest.mock import patch, MagicMock
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.core.config import get_settings


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_chat_endpoint_valid_request():
    mock_final_state = {
        "question": "How many days of annual leave do employees receive?",
        "search_query": "How many days of annual leave do employees receive?",
        "route": "kb",
        "kb_docs": [],
        "kb_grade": "GOOD",
        "web_results": [],
        "web_grade": None,
        "answer": "PeoplePrime employees receive 25 days of annual leave.",
        "citations": [{"title": "Leave_Policy.pdf (Page 2)", "url": "", "type": "private_kb", "page": 2}],
        "source_used": "private_kb",
        "retry_count": 0,
        "execution_trace": ["Router -> KB", "Pinecone retrieval -> 2 chunks", "KB grade -> GOOD"],
        "error_status": None,
    }

    with patch("app.api.routes.run_agentic_workflow", return_value=mock_final_state), \
         patch("app.api.routes.record_audit_entry") as mock_audit:

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/chat", json={"question": "How many days of annual leave do employees receive?"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["answer"] == "PeoplePrime employees receive 25 days of annual leave."
            assert data["source_used"] == "private_kb"
            assert data["kb_grade"] == "GOOD"
            assert data["retry_count"] == 0
            assert len(data["citations"]) == 1
            assert len(data["trace"]) == 3
            assert mock_audit.called


@pytest.mark.anyio
async def test_chat_endpoint_short_question():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/chat", json={"question": "a"})
        assert resp.status_code == 422 or resp.status_code == 400


@pytest.mark.anyio
async def test_chat_endpoint_empty_question():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/chat", json={"question": ""})
        assert resp.status_code == 422 or resp.status_code == 400
