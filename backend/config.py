from __future__ import annotations
import os
from pydantic import BaseModel

class Settings(BaseModel):
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "8080"))
    db_url: str = os.getenv("DB_URL", "sqlite:///./data/face_auth.db")
    thresh_auth: float = float(os.getenv("THRESH_AUTH", "0.88"))
    thresh_id: float = float(os.getenv("THRESH_ID", "0.90"))
    max_emb_per_user: int = int(os.getenv("MAX_EMB_PER_USER", "10"))
    img_size: int = int(os.getenv("IMG_SIZE", "160"))
    q_min_laplace: float = float(os.getenv("QUALITY_MIN_LAPLACE", "120.0"))
    q_brightness_min: int = int(os.getenv("QUALITY_BRIGHTNESS_MIN", "60"))
    q_brightness_max: int = int(os.getenv("QUALITY_BRIGHTNESS_MAX", "190"))
    q_max_tilt_deg: int = int(os.getenv("QUALITY_MAX_TILT_DEG", "15"))
    liveness_enabled: bool = os.getenv("LIVENESS_ENABLED", "false").lower() == "true"
    torch_num_threads: int = int(os.getenv("TORCH_NUM_THREADS", "1"))
    allowed_origins: str = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    debug_persist_images: bool = os.getenv("DEBUG_PERSIST_IMAGES", "false").lower() == "true"
settings = Settings()