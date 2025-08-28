from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session

from .db_models import FaceEmbedding, User

@dataclass
class IndexState:
    all_vecs: np.ndarray # shape [N, 512]
    owner_ids: np.ndarray # shape [N]
    by_user: Dict[int, np.ndarray] # user_id -> indices into all_vecs

def load_all(db: Session) -> IndexState:
    rows = (
        db.query(FaceEmbedding.id, FaceEmbedding.user_id, FaceEmbedding.vec)
        .order_by(FaceEmbedding.user_id)
        .all()
    )
    if not rows:
        return IndexState(np.zeros((0, 512), dtype=np.float32), np.zeros((0,), dtype=np.int32), {})
    vecs = []
    owners = []
    by_user: Dict[int, List[int]] = {}
    for i, r in enumerate(rows):
        v = np.frombuffer(r.vec, dtype=np.float32)
        vecs.append(v)
        owners.append(r.user_id)
        by_user.setdefault(r.user_id, []).append(i)
    all_vecs = np.vstack(vecs).astype(np.float32)
    owner_ids = np.asarray(owners, dtype=np.int32)
    by_user_np = {uid: np.asarray(ix, dtype=np.int32) for uid, ix in by_user.items()}
    return IndexState(all_vecs, owner_ids, by_user_np)

def reload_user(db: Session, user_id: int, state: IndexState) -> IndexState:
    # Minimal: rebuild all for simplicity (small N). Could optimize later.
    return load_all(db)

def nearest(vec: np.ndarray, state: IndexState) -> Tuple[Optional[int], float]:
    if state.all_vecs.shape[0] == 0:
        return None, 0.0
    scores = state.all_vecs @ vec # cosine if normalized
    i = int(np.argmax(scores))
    return int(state.owner_ids[i]), float(scores[i])

def auth_compare(user_id: int, vec: np.ndarray, state: IndexState) -> float:
    ix = state.by_user.get(user_id)
    if ix is None or ix.size == 0:
        return 0.0
    # max cosine to that user's enrolled vectors
    sub = state.all_vecs[ix]
    return float(np.max(sub @ vec))