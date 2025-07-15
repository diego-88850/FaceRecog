from pydantic import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite:///./face_auth.db"
    face_detection_model: str = "haarcascade_frontalface_default.xml"
    embedding_threshold: float = 0.85
    max_faces_per_user: int = 5

    class Config:
        env_file = ".env"