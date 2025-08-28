from __future__ import annotations
import hashlib
import io
from typing import Iterable
import numpy as np

def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v) + 1e-9)
    return v / n

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b)) # assume already L2-normalized

def cosine_batch(q: np.ndarray, M: np.ndarray) -> np.ndarray:
    return (M @ q).astype(np.float32) # [N]

def sha1_seed_from_bytes(b: bytes) -> int:
    h = hashlib.sha1(b).hexdigest()[:8]
    return int(h, 16)