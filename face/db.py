from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from config import Settings

# 1) Create the Engine
engine = create_engine(Settings.database_url, echo=True)

# Session factory pattern for dependency injection
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()