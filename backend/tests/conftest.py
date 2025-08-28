# backend/tests/conftest.py
import os, pytest

@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch):
    # read env at runtime so exports affect this
    USE_FAKE_EMBED  = os.getenv("USE_FAKE_EMBED", "1") == "1"
    USE_FAKE_DETECT = os.getenv("USE_FAKE_DETECT", "1") == "1"

    import backend.face as face
    if USE_FAKE_EMBED:
        def _fake_embed(self, img_bgr_160):
            import cv2, numpy as np
            from backend.utils import sha1_seed_from_bytes
            small = cv2.resize(img_bgr_160, (16, 16))
            seed = sha1_seed_from_bytes(small.tobytes())
            rng = np.random.RandomState(seed)
            v = rng.randn(512).astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-9)
            return v
        monkeypatch.setattr(face.FaceService, "embed", _fake_embed)

    if USE_FAKE_DETECT:
        def _fake_detect(self, img_bgr):
            h, w = img_bgr.shape[:2]
            return (w//4, h//4, w//2, h//2), None
        monkeypatch.setattr(face.FaceService, "detect_and_landmarks", _fake_detect)
