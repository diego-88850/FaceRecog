# Face Auth MVP — FastAPI + FaceNet (CPU‑only)

![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

A minimal, privacy‑friendly face **authentication & identification** service. Users enroll with a few photos; the service turns faces into compact embeddings and verifies future snapshots with cosine similarity. It’s CPU‑only, ships with fast deterministic tests, and includes an opt‑in “real model” test that runs MediaPipe + FaceNet locally.

---

## What it does (high level)
- **Detect & align:** Finds a face and eye landmarks (MediaPipe), upright‑crops to a consistent square (default **160×160**).
- **Embed:** Generates a **512‑D** L2‑normalized vector using **FaceNet (Inception‑ResNet v1, facenet‑pytorch, VGGFace2 weights)**.
- **Compare:** Uses **cosine similarity** (dot product on normalized vectors). Thresholds gate decisions for **auth** (verify a claimed user) or **id** (find the closest user).
- **Store:** Saves **embeddings only** in SQLite (not raw photos), and keeps an in‑memory index for fast lookup.
- **API:** `POST /register` to enroll photos, `POST /verify` to authenticate/identify, `GET /healthz` for health.

### Architecture at a glance
```mermaid
flowchart LR
  subgraph Client
    A[Enroll / Verify Request]\n(images + form-data)
  end
  subgraph Server(FastAPI)
    IO[image_io.py\nEXIF-safe decode]
    D[MediaPipe\nDetect + Landmarks]
    AL[Align 160×160]
    FN[FaceNet\n(Inception-ResNet v1)]
    DB[(SQLite\nEmbeddings only)]
  end
  A --> IO --> D --> AL --> FN
  FN -->|auth/id compare| DB
  DB -->|decision + score| A
```

---

## Repository layout
```
backend/
  app.py                 # FastAPI app: /register, /verify, /healthz
  face.py                # FaceService: detect → align → embed
  config.py              # Settings (Pydantic) from env vars
  image_io.py            # EXIF-safe bytes → BGR decode (shared by endpoints)
  scripts/
    sanity.py            # cosine sanity check on two images
  tests/
    conftest.py          # Test fakes (toggle via env), shared fixtures
    test_index.py        # Math/index unit tests (fake embeddings)
    test_api_flow.py     # API flow test (fake by default)
    test_api_flow_real.py# Real-model API test (opt-in)
    test_decode_consistency.py  # Deterministic decode test
    assets/
      pratt1.jpg
      pratt2.jpg
```

---

## Model details
- **Backbone**: `facenet-pytorch` → `InceptionResnetV1(pretrained='vggface2')`
- **Output**: 512‑D float embedding, **L2 normalized**
- **Preprocessing**:
  1. **Decode** upload bytes via `backend/image_io.py` → applies **EXIF rotation**, forces RGB → BGR for OpenCV/MediaPipe.
  2. **Detect & landmarks** with MediaPipe.
  3. **Align** to `IMG_SIZE` (default **160**), using eye landmarks (resize fallback if landmarks missing).
  4. **Embed** with FaceNet; compare with cosine.
- **Decisions & thresholds**:
  - `THRESH_AUTH` (default **0.88**) for “auth” mode
  - `THRESH_ID` (default **0.90**) for “id” mode

---

##  Installation
```bash
# 1) Create & activate venv
python3 -m venv .venv
source .venv/bin/activate

# 2) Install deps
python -m pip install -r requirements.txt

# If Pillow isn’t already listed
python -m pip install Pillow
```

---

## Run the API
```bash
# Environment (defaults shown)
export APP_HOST=0.0.0.0
export APP_PORT=8080
export DB_URL="sqlite:///./data/face_auth.db"
export THRESH_AUTH=0.88
export THRESH_ID=0.90
export IMG_SIZE=160
export QUALITY_MIN_LAPLACE=120.0
export QUALITY_BRIGHTNESS_MIN=60
export QUALITY_BRIGHTNESS_MAX=190
export QUALITY_MAX_TILT_DEG=15
export TORCH_NUM_THREADS=1
# Optional: reduce GPU/TF logs on macOS
export MEDIAPIPE_DISABLE_GPU=1
export TF_CPP_MIN_LOG_LEVEL=2

# Start
uvicorn backend.app:app --host $APP_HOST --port $APP_PORT --reload
```

### Example requests (cURL)
**Enroll**
```bash
curl -s -X POST "http://localhost:8080/register" \
  -F "username=alice" \
  -F "images=@/path/enroll1.jpg" -F "images=@/path/enroll2.jpg"
```

**Verify (auth)**
```bash
curl -s -X POST "http://localhost:8080/verify" \
  -F "mode=auth" -F "username=alice" \
  -F "image=@/path/verify.jpg"
```

**Identify (who is this?)**
```bash
curl -s -X POST "http://localhost:8080/verify" \
  -F "mode=id" \
  -F "image=@/path/unknown.jpg"
```

---

## 🧪 Testing
### Quick run (deterministic; great for CI)
By default, tests use **fake detection/embedding** so they’re fast & stable.
```bash
python -m pytest -q
```
How the fakes work (see `backend/tests/conftest.py`):
- `FaceService.embed` is monkeypatched to return a deterministic unit vector from image bytes.
- `FaceService.detect_and_landmarks` can be patched to always return a center box.
- Toggled by env vars (below).

### Real‑model test (MediaPipe + FaceNet)
Run a realistic test locally (slower; uses actual detection/embedding):
```bash
# Turn OFF fakes
export USE_FAKE_EMBED=0
export USE_FAKE_DETECT=0
# Nice-to-haves for macOS + quieter logs
export MEDIAPIPE_DISABLE_GPU=1
export TF_CPP_MIN_LOG_LEVEL=2
# Use a separate DB so test data is isolated
export DB_URL="sqlite:///./data/test_face_auth.db"

# Run only the real-model tests
python -m pytest -q -m realmodel -s
```
Mark real tests:
```python
import pytest
@pytest.mark.realmodel
def test_register_and_verify_real():
    ...
```
Register the marker in **pytest.ini**:
```ini
[pytest]
markers =
    realmodel: runs tests that require the real detector/FaceNet (slower)
```

### Sanity script (CLI)
Quick cosine check on two images:
```bash
python -m backend.scripts.sanity /path/a.jpg /path/b.jpg
# prints: cosine=0.8342 (etc). Same file twice ≈ 1.0000
```

### Decode determinism test
```bash
python -m pytest -q backend/tests/test_decode_consistency.py
```

---

## Environment variables

| Variable | Default | Purpose |
|---|---:|---|
| `APP_HOST` | `0.0.0.0` | Uvicorn bind host |
| `APP_PORT` | `8080` | Uvicorn port |
| `DB_URL` | `sqlite:///./data/face_auth.db` | SQLite database URL |
| `THRESH_AUTH` | `0.88` | Accept threshold for auth |
| `THRESH_ID` | `0.90` | Accept threshold for id |
| `IMG_SIZE` | `160` | Aligned crop size |
| `QUALITY_MIN_LAPLACE` | `120.0` | Blur/edge strength minimum |
| `QUALITY_BRIGHTNESS_MIN` | `60` | Minimum brightness |
| `QUALITY_BRIGHTNESS_MAX` | `190` | Maximum brightness |
| `QUALITY_MAX_TILT_DEG` | `15` | Max tilt angle |
| `TORCH_NUM_THREADS` | `1` | Torch CPU threads |
| `DEBUG_PERSIST_IMAGES` | `false` | Save aligned crops for debugging |
| `USE_FAKE_EMBED` | `1` (tests) | Toggle fake embedder in tests |
| `USE_FAKE_DETECT` | `1` (tests) | Toggle fake detector in tests |
| `MEDIAPIPE_DISABLE_GPU` | *(unset)* | Set `1` to disable GPU on macOS |
| `TF_CPP_MIN_LOG_LEVEL` | *(unset)* | Set `2` to quiet TF logs |

---

## Implementation notes 
- **422 on no face**: API returns `422 {"detail":"NO_FACE"}` when detection fails (by design).
- **“LOW_CONFIDENCE”**: returned when cosine < threshold. If this happens on real photos:
  - Ensure unified decode is used in both endpoints.
  - Try `MEDIAPIPE_DISABLE_GPU=1` to reduce tiny landmark jitter on macOS.
  - Enroll 5–10 varied shots per user; avoid near duplicates.
- **Privacy**: by default **no raw images are persisted**. Use `DEBUG_PERSIST_IMAGES=true` only for debugging (saves aligned 160×160 crops).
- **Performance**: keep `TORCH_NUM_THREADS=1` (or 2) for predictable CPU latency; optionally warm the model on startup.

---

##  Roadmap / future work
- **Liveness** (blink/EAR or micro‑motion), basic anti‑spoof checks.
- **Vector index** (FAISS) for faster global ID on large user sets.
- **Rate limiting & auth** (JWT/API keys), richer audit logs.
- **Storage backends** (Postgres, cloud vector DB), encryption at rest.
- **Metrics UI** (score histograms, ROC tuning, thresholds per cohort).
- **Alternate detectors** (RetinaFace/YOLOv8‑face) as drop‑ins to `FaceService`.
- **Edge build** (ONNX/NCNN) for low‑resource deployment.

---

## Acknowledgements
- **FaceNet** via [`facenet-pytorch`](https://github.com/timesler/facenet-pytorch) (Inception‑ResNet v1, VGGFace2 weights)
- **MediaPipe** Face Mesh/Detection for lightweight landmarking
- **FastAPI**, **Pydantic**, **SQLAlchemy** for the service core

---

## License
MIT (or your choice). Ensure compliance with `facenet-pytorch`, MediaPipe, and model‑weight licenses.

