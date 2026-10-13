from unittest.mock import patch
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.core.config import get_settings


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_ingest_unauthorized():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        files = {"file": ("policy.txt", b"Some policy text", "text/plain")}
        res = await client.post("/api/documents/ingest", files=files, headers={"x-admin-key": "invalid-key"})
        assert res.status_code == 401
        assert "Unauthorized" in res.json()["detail"]


@pytest.mark.anyio
async def test_ingest_missing_header():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        files = {"file": ("policy.txt", b"Some policy text", "text/plain")}
        res = await client.post("/api/documents/ingest", files=files)
        assert res.status_code == 401


@pytest.mark.anyio
async def test_ingest_unsupported_file_type():
    settings = get_settings()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        files = {"file": ("malicious.exe", b"binary content", "application/octet-stream")}
        res = await client.post(
            "/api/documents/ingest",
            files=files,
            headers={"x-admin-key": settings.admin_api_key},
        )
        assert res.status_code == 400
        assert "Unsupported file type" in res.json()["detail"]


@pytest.mark.anyio
async def test_ingest_empty_file():
    settings = get_settings()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        files = {"file": ("empty.txt", b"", "text/plain")}
        res = await client.post(
            "/api/documents/ingest",
            files=files,
            headers={"x-admin-key": settings.admin_api_key},
        )
        assert res.status_code == 400
        assert "empty" in res.json()["detail"].lower()


@pytest.mark.anyio
async def test_ingest_oversized_file():
    settings = get_settings()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Patch max_upload_size_bytes to a very small threshold for testing
        with patch.object(settings, "max_upload_size_bytes", 100):
            files = {"file": ("large.txt", b"A" * 200, "text/plain")}
            res = await client.post(
                "/api/documents/ingest",
                files=files,
                headers={"x-admin-key": settings.admin_api_key},
            )
            assert res.status_code == 413
            assert "exceeds maximum size" in res.json()["detail"]


@pytest.mark.anyio
async def test_ingest_success_mocked_pinecone():
    settings = get_settings()
    policy_content = (
        "PeoplePrime Code of Conduct: Mutual respect, anti-harassment, and confidentiality "
        "are core requirements for all employees and contractors."
    )

    with patch("app.api.routes.upsert_chunks", return_value=1) as mock_upsert:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            files = {"file": ("code_of_conduct.txt", policy_content.encode("utf-8"), "text/plain")}
            res = await client.post(
                "/api/documents/ingest",
                files=files,
                headers={"x-admin-key": settings.admin_api_key},
            )

            assert res.status_code == 200
            data = res.json()
            assert data["message"] == "Document successfully processed and indexed into Pinecone."
            assert data["file"] == "code_of_conduct.txt"
            assert data["chunks"] == 1
            assert data["dimension"] == 384
            assert mock_upsert.called


@pytest.mark.anyio
async def test_ingest_alias_compatibility():
    settings = get_settings()
    with patch("app.api.routes.upsert_chunks", return_value=1):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            files = {"file": ("handbook.md", b"# Handbook\nContent here.", "text/markdown")}
            res = await client.post(
                "/api/ingest",
                files=files,
                headers={"x-admin-key": settings.admin_api_key},
            )
            assert res.status_code == 200
            assert res.json()["chunks"] == 1
