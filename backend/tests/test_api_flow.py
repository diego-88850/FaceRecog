# backend/tests/test_api_flow_real.py
import os
# Turn OFF test fakes and quiet logs
os.environ["USE_FAKE_EMBED"]  = "0"
os.environ["USE_FAKE_DETECT"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"   # optional: less jitter on macOS
os.environ["DB_URL"] = "sqlite:///./data/test_face_auth.db"

# IMPORTANT: reload config & app so env vars take effect
import importlib
import backend.config as config
importlib.reload(config)
import backend.app as app_module
importlib.reload(app_module)
from backend.app import app

from pathlib import Path
from fastapi.testclient import TestClient
import pytest

ASSETS = Path(__file__).parent / "assets"

def _img_bytes(name: str) -> bytes:
    return (ASSETS / name).read_bytes()

@pytest.mark.realmodel
def test_register_and_verify_real():
    c = TestClient(app)

    # 1) Enroll ONE image
    files = [
        ("images", ("enroll1.jpg", _img_bytes("pratt1.jpg"), "image/jpeg")),
    ]
    r = c.post("/register", data={"username": "Pratt"}, files=files)
    assert r.status_code == 200, r.text

    # 2) Verify with the EXACT SAME bytes
    r2 = c.post(
        "/verify",
        data={"mode": "auth", "username": "Pratt"},
        files={"image": ("verify.jpg", _img_bytes("pratt1.jpg"), "image/jpeg")},
    )
    assert r2.status_code == 200, r2.text
    body = r2.json()
    print("BODY:", body)
    assert body["decision"] in ("ACCEPT", "MATCH", "OK"), body
