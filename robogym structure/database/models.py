
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()
engine = create_engine('sqlite:///robogym.db')
SessionLocal = sessionmaker(bind=engine)

class TrainingLog(Base):
    __tablename__ = 'training_logs'
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String)
    reward = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)
