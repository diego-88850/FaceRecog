# backend/scripts/sanity.py
import sys, time
import numpy as np
from backend.face import FaceService
from backend.image_io import decode_image_bytes_to_bgr  # <-- NEW
from pathlib import Path

def load_bgr(path: str):
    data = Path(path).read_bytes()
    return decode_image_bytes_to_bgr(data)  # EXIF-safe decode

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python -m backend.scripts.sanity imgA.jpg imgB.jpg")
        sys.exit(1)

    imgA = load_bgr(sys.argv[1])
    imgB = load_bgr(sys.argv[2])

    svc = FaceService()

    def align_and_embed(img_bgr, tag):
        box, eyes = svc.detect_and_landmarks(img_bgr)
        if box is None:
            print(f"[{tag}] NO_FACE")
            sys.exit(2)
        aligned = svc.align(img_bgr, eyes, 160)
        # save crops to compare visually
        import cv2
        cv2.imwrite(f"aligned_{tag}.png", aligned)
        t0 = time.perf_counter()
        v = svc.embed(aligned)
        t1 = time.perf_counter()
        print(f"[{tag}] embed_ms={(t1 - t0)*1000:.2f}")
        # sanity: L2 norm
        print(f"[{tag}] norm={np.linalg.norm(v):.6f}")
        return v

    vA = align_and_embed(imgA, "A")
    vB = align_and_embed(imgB, "B")
    sim = float(np.dot(vA/np.linalg.norm(vA), vB/np.linalg.norm(vB)))
    print(f"cosine={sim:.4f}")
