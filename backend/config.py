from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8080

    DB_URL: str = "sqlite:///./data/face_auth.db"

    THRESH_AUTH: float = 0.88
    THRESH_ID: float = 0.90
    MAX_EMB_PER_USER: int = 10

    IMG_SIZE: int = 160

    QUALITY_MIN_LAPLACE: float = 120.0
    QUALITY_BRIGHTNESS_MIN: int = 60
    QUALITY_BRIGHTNESS_MAX: int = 190
    QUALITY_MAX_TILT_DEG: int = 15

    LIVENESS_ENABLED: bool = False
    TORCH_NUM_THREADS: int = 1

    ALLOWED_ORIGINS: List[str] = ["http://localhost:5173"]
    LOG_LEVEL: str = "INFO"
    DEBUG_PERSIST_IMAGES: bool = False

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

settings = Settings()
