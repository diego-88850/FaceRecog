from __future__ import annotations
import io
import json
import os
import time
from typing import List, Optional
import numpy as np
import cv2
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from .config import settings
from .db import get_db, SessionLocal
from .db_models import Attempt, Base, FaceEmbedding, User
from .index import IndexState, auth_compare, load_all, nearest, reload_user
from .logger import log_middleware
from .schemas import VerifyResponse
from .utils import l2_normalize
from . import face as face_mod
from backend.face import interocular_fraction, MIN_EYE_FRAC, MAX_EYE_FRAC

app = FastAPI()

# DB init (SQLite)
Base.metadata.create_all(bind=SessionLocal.kw['bind'])

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.allowed_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Logging
app.middleware("http")(log_middleware)

# Global state
STATE: IndexState = load_all(SessionLocal())
FACE = face_mod.FaceService(img_size=settings.img_size,
torch_num_threads=settings.torch_num_threads)

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/register")
def register(
    username: str = Form(...),
    images: List[UploadFile] = File(...),
    request: Request = None,
    db=Depends(get_db),
):
    t0 = time.perf_counter()
    if not images:
        raise HTTPException(status_code=422, detail="No images provided")
    user = db.query(User).filter(User.username == username).first()
    if not user:
        user = User(username=username)
        db.add(user)
        db.commit()
        db.refresh(user)

    # Load existing vecs for dedupe
    existing = db.query(FaceEmbedding).filter(FaceEmbedding.user_id == user.id).all()
    existing_vecs = [l2_normalize(np.frombuffer(e.vec, dtype=np.float32)) for e in existing]

    stored = 0
    rejected = {"quality": 0, "duplicate": 0, "noface": 0}
    new_vecs: List[np.ndarray] = []

    for f in images[: settings.max_emb_per_user * 2]:  # soft cap work
        content = f.file.read()
        data = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            rejected["noface"] += 1
            continue
        box, eyes = FACE.detect_and_landmarks(img)
        if box is None:
            rejected["noface"] += 1
            continue

        ok, reason, eye_frac = FACE.check_scale(img, eyes)
        if not ok:
            raise HTTPException(status_code=422, detail=reason)

        aligned = FACE.align(img, eyes, settings.img_size)
        q = FACE.quality_report(aligned, eyes)
        if q["laplace_var"] < settings.q_min_laplace or not (settings.q_brightness_min <= q["brightness"] <= settings.q_brightness_max) or q["tilt_deg"] > settings.q_max_tilt_deg:
            rejected["quality"] += 1
            continue

        vec = FACE.embed(aligned)
        vec = l2_normalize(vec)
        # dedupe against existing and new
        def is_dup(v: np.ndarray, pool: List[np.ndarray]) -> bool:
            if not pool:
                return False
            M = np.vstack(pool)
            scores = M @ v
            return bool(np.max(scores) > 0.995)

        if is_dup(vec, existing_vecs) or is_dup(vec, new_vecs):
            rejected["duplicate"] += 1
            continue
        new_vecs.append(vec)

    # persist up to MAX_EMB_PER_USER
    to_store = new_vecs[: max(0, settings.max_emb_per_user - len(existing_vecs))]
    for v in to_store:
        db.add(FaceEmbedding(user_id=user.id, vec=v.astype(np.float32).tobytes(), norm=float(np.linalg.norm(v))))
    db.commit()
    global STATE
    STATE = reload_user(db, user.id, STATE)
    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "username": username,
        "stored_embeddings": len(to_store),
        "rejected": rejected,
        "latency_ms": round(latency_ms, 2),
    }

@app.post("/verify", response_model=VerifyResponse)
async def verify(
    mode: str = Form(...),
    image: UploadFile = File(...),
    username: Optional[str] = Form(None),
    request: Request = None,
    db=Depends(get_db),
):
    if mode not in ("auth", "id"):
        raise HTTPException(status_code=400, detail="mode must be 'auth' or 'id'")
    t0 = time.perf_counter()
    content = await image.read()
    data = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=422, detail="Invalid image")

    box, eyes = FACE.detect_and_landmarks(img)
    if box is None or eyes is None:
        raise HTTPException(status_code=422, detail="NO_FACE")

    aligned = FACE.align(img, eyes, settings.img_size)
    vec = FACE.embed(aligned)
    vec = l2_normalize(vec)
    decision = "UNKNOWN"
    who = None
    confidence = 0.0
    matched_user_id = None
    if mode == "auth":
        if not username:
            raise HTTPException(status_code=400, detail="username required for auth mode")
        db_user = db.query(User).filter(User.username == username).first()
        if not db_user:
            decision = "UNKNOWN_USER"
        else:
            score = auth_compare(db_user.id, vec, STATE)
            confidence = score
            if score >= settings.thresh_auth:
                decision = "ACCEPT"
                who = db_user.username
                matched_user_id = db_user.id
            else:
                decision = "LOW_CONFIDENCE"
                who = db_user.username
    else:  # id mode
        uid, score = nearest(vec, STATE)
        confidence = score
        if uid is not None and score >= settings.thresh_id:
            u = db.query(User).get(uid)
            decision = "ACCEPT"
            who = u.username if u else None
            matched_user_id = uid
        else:
            decision = "UNKNOWN"

    latency_ms = (time.perf_counter() - t0) * 1000
    # log attempt
    db.add(Attempt(
        mode=mode,
        user_hint=username,
        matched_user_id=matched_user_id,
        score=confidence,
        passed=1 if decision == "ACCEPT" else 0,
        reason=decision,
        latency_ms=float(latency_ms),
        client_meta={"filename": image.filename},
    ))
    db.commit()

    return VerifyResponse(decision=decision, who=who, confidence=round(confidence, 4), latency_ms=round(latency_ms, 2))

@app.get("/metrics-lite")
def metrics_lite(db=Depends(get_db)):
    n = db.query(Attempt).count()
    latencies = [x[0] for x in db.query(Attempt.latency_ms).all()]
    avg = sum(latencies) / len(latencies) if latencies else 0.0
    return {"attempts": n, "avg_latency_ms": round(avg, 2)}

# Optionally serve frontend (built) if present
if os.path.isdir("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")