from sqlalchemy import Column, Integer, String, Float, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# SQLite DB file
DB_PATH = "robogym.db"
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

# Setup
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --------------------------
# Database Table Definitions
# --------------------------

class TrainingLog(Base):
    __tablename__ = "training_logs"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True)
    mean_reward = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

# --------------------------
# Utility Functions
# --------------------------

def init_db():
    """Create tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def log_training(model_name: str, mean_reward: float):
    """Insert a training log."""
    session = SessionLocal()
    log = TrainingLog(model_name=model_name, mean_reward=mean_reward)
    session.add(log)
    session.commit()
    session.close()



def fetch_logs(model_name: str = None):
    """Fetch logs, optionally filtered by model."""
    session = SessionLocal()
    query = session.query(TrainingLog)
    if model_name:
        query = query.filter(TrainingLog.model_name == model_name)
    logs = query.all()
    session.close()
    return logs
