from pathlib import Path
from backend.image_io import decode_image_bytes_to_bgr

def test_decode_same_bytes_identical_array():
    p = Path(__file__).parent / "assets" / "pratt1.jpg"
    data = p.read_bytes()
    a = decode_image_bytes_to_bgr(data)
    b = decode_image_bytes_to_bgr(data)
    assert a.shape == b.shape and (a == b).all()
