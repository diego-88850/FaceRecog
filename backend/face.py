from __future__ import annotations
import io
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np

TARGET_SIZE = 160
TARGET_EYE_DIST = 64
EYES_Y = 60
MARGIN_SCALE = 1.35
MIN_EYE_FRAC = 0.14
MAX_EYE_FRAC = 0.55

def interocular_fraction(eyes: "Eyes", img_shape) -> float:
    """
    Fraction = interocular distance / min(H, W) in the *input* image.
    """
    (lx, ly), (rx, ry) = eyes.left, eyes.right
    d = float(np.hypot(rx - lx, ry - ly))
    short_side = float(min(img_shape[0], img_shape[1]))
    return d / max(short_side, 1.0)

def _center_resize(img_bgr: np.ndarray, out_size: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    s = min(h, w)
    y0 = max(0, (h - s) // 2)
    x0 = max(0, (w - s) // 2)
    return cv2.resize(img_bgr[y0:y0 + s, x0:x0 + s], (out_size, out_size))

# Lazy imports for heavy deps
try:
    import mediapipe as mp  # type: ignore
except Exception:  # pragma: no cover
    mp = None  # fallback later

try:
    from facenet_pytorch import InceptionResnetV1 # type: ignore
    _FACENET_AVAILABLE = True
except Exception: # pragma: no cover
    _FACENET_AVAILABLE = False

@dataclass
class Eyes:
    left: Tuple[float, float]
    right: Tuple[float, float]

class FaceService:
    def __init__(self, img_size: int = 160, torch_num_threads: int = 1):
        self.img_size = img_size
        self.resnet = None
        self._init_embedder(torch_num_threads)
        self._init_detector()

    def _init_embedder(self, threads: int):
        if _FACENET_AVAILABLE:
            import torch
            torch.set_num_threads(threads)
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        else:
            self.resnet = None  # tests will monkeypatch embed()

    def _init_detector(self):
        self._mp_face_mesh = None
        if mp is not None:
            self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

    def detect_and_landmarks(self, img_bgr: np.ndarray) -> Tuple[Optional[Tuple[int,int,int,int]], Optional[Eyes]]:
        h, w = img_bgr.shape[:2]
        if self._mp_face_mesh is None:
            # naive fallback: assume center box; no eye landmarks
            box = (w // 4, h // 4, w // 2, h // 2)
            return box, None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        out = self._mp_face_mesh.process(img_rgb)
        if not out.multi_face_landmarks:
            return None, None
        lm = out.multi_face_landmarks[0].landmark
        # eyes: use approximate indices
        def _avg(ids: List[int]):
            xs = [lm[i].x for i in ids]
            ys = [lm[i].y for i in ids]
            return (float(np.mean(xs) * w), float(np.mean(ys) * h))

        left_ids = [33, 133, 160, 159]
        right_ids = [362, 263, 387, 386]
        left = _avg(left_ids)
        right = _avg(right_ids)
        x0 = max(0, int(min(left[0], right[0]) - 0.4 * self.img_size))
        y0 = max(0, int(min(left[1], right[1]) - 0.6 * self.img_size))
        x1 = min(w, int(max(left[0], right[0]) + 0.4 * self.img_size))
        y1 = min(h, int(max(left[1], right[1]) + 0.8 * self.img_size))
        box = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))
        return box, Eyes(left=left, right=right)

    def align(self, img_bgr, eyes, out_size: int = TARGET_SIZE):
        """
        Align to a canonical face: rotate so eyes are horizontal, scale so eye distance is fixed.
        `eyes` may be an Eyes dataclass or ((lx,ly),(rx,ry)).
        """
        # 0) Extract eye coords in a tolerant way
        if eyes is None:
            return _center_resize(img_bgr, out_size)

        if isinstance(eyes, Eyes):
            lx, ly = eyes.left
            rx, ry = eyes.right
        else:
            (lx, ly), (rx, ry) = eyes

        # 1) Fallback immediately if anything is missing or NaN
        vals = (lx, ly, rx, ry)
        if any(v is None for v in vals):
            return _center_resize(img_bgr, out_size)
        if any(not np.isfinite(v) for v in vals):
            return _center_resize(img_bgr, out_size)

        # 2) Rotate so eyes are horizontal
        dx, dy = (rx - lx), (ry - ly)
        eye_dist = float(np.hypot(dx, dy))
        if eye_dist < 1e-6:  # degenerate
            return _center_resize(img_bgr, out_size)

        angle = np.degrees(np.arctan2(dy, dx))
        center = (0.5 * (lx + rx), 0.5 * (ly + ry))
        Mrot = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot = cv2.warpAffine(
            img_bgr, Mrot, (img_bgr.shape[1], img_bgr.shape[0]),
            flags=cv2.INTER_LINEAR
        )

        # 3) Recompute rotated eye coords
        def warp_pt(M, x, y):
            v = np.array([x, y, 1.0], dtype=np.float32)
            xr, yr = np.dot(M, v)
            return float(xr), float(yr)

        lx_r, ly_r = warp_pt(Mrot, lx, ly)
        rx_r, ry_r = warp_pt(Mrot, rx, ry)

        # 4) Scale so inter-ocular distance matches target, then translate
        eye_dist_r = float(np.hypot(rx_r - lx_r, ry_r - ly_r))
        if eye_dist_r < 1e-6:
            return _center_resize(img_bgr, out_size)

        scale = TARGET_EYE_DIST / eye_dist_r
        cx = 0.5 * (lx_r + rx_r)
        cy = 0.5 * (Ly := ly_r + ry_r) / 2.0  # avoid typos; or just 0.5*(ly_r+ry_r)

        M = np.array(
            [[scale, 0.0, out_size / 2.0 - scale * cx],
             [0.0, scale, EYES_Y         - scale * cy]],
            dtype=np.float32
        )

        aligned = cv2.warpAffine(
            rot, M, (out_size, out_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        return aligned

    def quality_report(self, img_bgr: np.ndarray, eyes: Optional[Eyes]) -> dict:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        mean = float(np.mean(gray))
        tilt = 0.0
        if eyes is not None:
            (lx, ly), (rx, ry) = eyes.left, eyes.right
            dy, dx = (ry - ly), (rx - lx)
            tilt = abs(float(np.degrees(np.arctan2(dy, dx))))
        eye_frac = interocular_fraction(eyes, img_bgr.shape) if eyes is not None else 0.0
        return {"laplace_var": float(lap), "brightness": mean, "tilt_deg": tilt, "eye_frac": eye_frac}

    def embed(self, img_bgr_160: np.ndarray) -> np.ndarray:
        if self.resnet is None:
            raise RuntimeError("Embedder not available; use fake in tests or install facenet-pytorch.")
        import torch
        img_rgb = cv2.cvtColor(img_bgr_160, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
        x = (x - 0.5) / 0.5
        x = x.unsqueeze(0)
        with torch.no_grad():
            vec = self.resnet(x).cpu().numpy()[0]
        vec = vec.astype(np.float32)
        vec /= (np.linalg.norm(vec) + 1e-9)
        return vec

    def check_scale(self, img_bgr: np.ndarray, eyes: Optional[Eyes]):
        """
        Returns (ok: bool, reason: Optional[str], eye_frac: float)
        """
        if eyes is None:
            return False, "NO_FACE", 0.0
        frac = interocular_fraction(eyes, img_bgr.shape)
        if frac < MIN_EYE_FRAC:
            return False, "FACE_TOO_SMALL", frac
        if frac > MAX_EYE_FRAC:
            return False, "FACE_TOO_CLOSE", frac
        return True, None, frac