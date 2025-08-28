# backend/image_io.py
import io
import numpy as np
from PIL import Image, ImageOps
import cv2

def decode_image_bytes_to_bgr(data: bytes) -> np.ndarray:
    """
    Bytes (jpg/png/etc) -> HxWx3 BGR uint8
    - Applies EXIF rotation
    - Forces 3-channel RGB then converts to BGR
    """
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")  # ensure upright, 3-ch
    arr = np.array(img)  # RGB uint8
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr

# **not really using this anymore but keeping just in case ever needed**