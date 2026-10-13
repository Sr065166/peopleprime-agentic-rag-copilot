import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.core.config import get_settings


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_health_endpoint():
    settings = get_settings()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == settings.app_name
        assert data["version"] == settings.app_version
        assert data["environment"] == settings.app_env


@pytest.mark.anyio
async def test_home_page():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/")
        assert response.status_code == 200
        assert "PeoplePrime Technologies" in response.text
        assert "HR Policy" in response.text
