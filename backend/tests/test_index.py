import pytest
from httpx import AsyncClient
from fastapi import status
from backend/app import app

@pytest.mark.anyio
async def test_healthz():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.get("/healthz")
    assert r.status_code == status.HTTP_200_OK
    assert r.json() == {"ok": True}
