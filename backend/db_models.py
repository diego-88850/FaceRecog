from __future__ import annotations
from datetime import datetime
from typing import Optional
from sqlalchemy import JSON, BLOB, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(128), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    embeddings = relationship("FaceEmbedding", back_populates="user", cascade="all, delete-orphan")

class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    vec = Column(BLOB, nullable=False)  # float32/16 bytes
    norm = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    user = relationship("User", back_populates="embeddings")

class Attempt(Base):
    __tablename__ = "attempts"
    id = Column(Integer, primary_key=True)
    ts = Column(DateTime, default=datetime.utcnow, nullable=False)
    mode = Column(String(16), nullable=False) # auth|id
    user_hint = Column(String(128))
    matched_user_id = Column(Integer)
    score = Column(Float)
    passed = Column(Integer) # 0/1
    reason = Column(String(64))
    latency_ms = Column(Float)
    client_meta = Column(JSON)
    server_meta = Column(JSON)
    extra = Column(Text)