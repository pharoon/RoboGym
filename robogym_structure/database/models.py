# models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, Enum, ForeignKey, create_engine, JSON,func
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timezone
from database.Enums import AlgorithmType,RoboticArmType
import enum
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()   



SQLALCHEMY_DATABASE_URL = f"postgresql://postgres.xljntaujspiljiczjzzh:3aCXHe0fDQx0gzNL@aws-0-eu-central-1.pooler.supabase.com:6543/postgres"
# SQLAlchemy setup
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    trained_models = relationship("TrainedModel", back_populates="user", cascade="all, delete-orphan")
    stats = relationship("UserStats", uselist=False, back_populates="user", cascade="all, delete-orphan")

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(str(self.password_hash), password)

class UserStats(Base):
    __tablename__ = "user_stats"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    tests_run = Column(Integer, default=0)
    train_sessions_count = Column(Integer, default=0)
    trained_models_count = Column(Integer, default=0)

    user = relationship("User", back_populates="stats")

class TrainedModel(Base):
    __tablename__ = "trained_models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    model_path = Column(String, nullable=False)
    algorithm = Column(Enum(AlgorithmType), nullable=False)
    robotic_arm = Column(Enum(RoboticArmType), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    total_timesteps = Column(Integer, default=0)
    total_training_time = Column(Float, default=0.0)
    final_mean_reward = Column(Float)

    user = relationship("User", back_populates="trained_models")
    train_sessions = relationship("TrainSession", back_populates="model", cascade="all, delete-orphan")

class TrainSession(Base):
    __tablename__ = "train_sessions"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("trained_models.id"), nullable=False)
    timesteps = Column(Integer, nullable=False)
    total_time = Column(Float, nullable=False)
    mean_reward = Column(Float)
    started_at = Column(DateTime(timezone=True), default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    train_log = Column(JSON, nullable=True)

    model = relationship("TrainedModel", back_populates="train_sessions")

# --------------------------
# Database Utility Functions
# --------------------------

def init_db():
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()
    print("✅ Tables created successfully.")
