# models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, Enum, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
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

# --------------------------
# Enum Types
# --------------------------

class AlgorithmType(enum.Enum):
    PPO = "PPO"
    DQN = "DQN"
    SAC = "SAC"
    TD3 = "TD3"

class RoboticArmType(enum.Enum):
    KUKA_IIWA = "kuka_iiwa"
    UR5 = "ur5"
    PANDA = "panda"

# --------------------------
# Database Table Definitions
# --------------------------

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    trained_models = relationship("TrainedModel", back_populates="user", cascade="all, delete-orphan")
    train_sessions = relationship("TrainSession", back_populates="user", cascade="all, delete-orphan")

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        # Convert SQLAlchemy Column to string
        stored_hash = str(self.password_hash) if self.password_hash is not None else ""
        return check_password_hash(stored_hash, password)

class TrainedModel(Base):
    __tablename__ = "trained_models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    model_path = Column(String, nullable=False)
    algorithm = Column(Enum(AlgorithmType), nullable=False)
    robotic_arm = Column(Enum(RoboticArmType), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    user = relationship("User", back_populates="trained_models")
    train_sessions = relationship("TrainSession", back_populates="model", cascade="all, delete-orphan")

class TrainingProgress(Base):
    __tablename__ = "training_progress"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("train_sessions.id"), nullable=False)
    timestep = Column(Integer, nullable=False)
    mean_reward = Column(Float)
    logged_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("TrainSession", back_populates="progress_logs")

class TrainSession(Base):
    __tablename__ = "train_sessions"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("trained_models.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    timesteps = Column(Integer, nullable=False)
    total_time = Column(Float, nullable=False)
    mean_reward = Column(Float)  # Final mean reward
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

    user = relationship("User", back_populates="train_sessions")
    model = relationship("TrainedModel", back_populates="train_sessions")
    progress_logs = relationship("TrainingProgress", back_populates="session", cascade="all, delete-orphan")

# --------------------------
# Database Utility Functions
# --------------------------

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# User utilities
def create_user(username: str, email: str, password: str):
    db = SessionLocal()
    try:
        user = User(username=username, email=email)
        user.set_password(password)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()

def get_user_by_username(username: str):
    db = SessionLocal()
    try:
        return db.query(User).filter(User.username == username).first()
    finally:
        db.close()
def get_user_by_email(email: str):
    db = SessionLocal()
    try:
        return db.query(User).filter(User.email == email).first()
    finally:
        db.close()

def get_user_by_id(user_id: int):
    db = SessionLocal()
    try:
        return db.query(User).filter(User.id == user_id).first()
    finally:
        db.close()


# TrainedModel utilities
def create_trained_model(name, model_path, algorithm, robotic_arm, user_id):
    db = SessionLocal()
    try:
        model = TrainedModel(
            name=name,
            model_path=model_path,
            algorithm=algorithm,
            robotic_arm=robotic_arm,
            user_id=user_id
        )
        db.add(model)
        db.commit()
        db.refresh(model)
        return model
    finally:
        db.close()

def get_user_models(user_id: int):
    db = SessionLocal()
    try:
        return db.query(TrainedModel).filter(TrainedModel.user_id == user_id).all()
    finally:
        db.close()

def get_trained_model_by_name_and_user(user_id: int, model_name: str):
    """
    Fetch a trained model by its name for a specific user.
    Returns the model if found, None otherwise.
    """
    db = SessionLocal()
    try:
        return (
            db.query(TrainedModel)
            .filter(TrainedModel.name == model_name)
            .filter(TrainedModel.user_id == user_id)
            .first()
        )
    finally:
        db.close()

def delete_trained_model(user_id: int, model_name: str) -> bool:
    """
    Delete a trained model for a specific user by model name.
    Returns True if deletion was successful, False otherwise.
    """
    db = SessionLocal()
    try:
        model = (
            db.query(TrainedModel)
            .filter(TrainedModel.name == model_name)
            .filter(TrainedModel.user_id == user_id)
            .first()
        )

        if not model:
            print(f"Model '{model_name}' not found for user ID {user_id}")
            return False

        db.delete(model)
        db.commit()
        return True

    except Exception as e:
        print(f"Error deleting model '{model_name}': {str(e)}")
        db.rollback()
        return False

    finally:
        db.close()

def updateModelPath(model_id: int, new_model_path: str):
    """Update the model path for a specific trained model."""
    db = SessionLocal()
    try:
        model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
        if not model:
            print(f"Model with ID {model_id} not found.")
            return False

        model.model_path = new_model_path
        db.commit()
        return True

    except Exception as e:
        print(f"Error updating model path for ID {model_id}: {str(e)}")
        db.rollback()
        return False

    finally:
        db.close()

def model_rename(user_id: int, old_name: str, new_name: str) -> bool:
    """
    Rename a trained model for a specific user.
    Returns True if the rename was successful, False otherwise.
    """
    if not new_name:
            print("New model name cannot be empty.")
            return False
    
    db = SessionLocal()
    try:
        model = (
            db.query(TrainedModel)
            .filter(TrainedModel.name == old_name)
            .filter(TrainedModel.user_id == user_id)
            .first()
        )

        if not model:
            print(f"Model '{old_name}' not found for user ID {user_id}")
            return False
        # Check if the new name already exists for this user
        existing_model = (
            db.query(TrainedModel)
            .filter(TrainedModel.name == new_name)
            .filter(TrainedModel.user_id == user_id)
            .first()
        )
        if existing_model:
            print(f"Model '{new_name}' already exists for user ID {user_id}")
            return False
        
        
        model.name = new_name
        db.commit()
        return True

    except Exception as e:
        print(f"Error renaming model '{old_name}': {str(e)}")
        db.rollback()
        return False

    finally:
        db.close()

     
# TrainSession utilities
def create_train_session(model_id, user_id, timesteps, total_time, mean_reward=None):
    """Create or complete a training session."""
    db = SessionLocal()
    try:
        # Find any existing incomplete session
        current_session = (
            db.query(TrainSession)
            .filter(TrainSession.model_id == model_id)
            .filter(TrainSession.completed_at.is_(None))
            .first()
        )

        now = datetime.utcnow()

        if current_session:
            # Update the existing session with final values
            current_session.timesteps = int(timesteps)
            current_session.total_time = float(total_time)
            if mean_reward is not None:
                current_session.mean_reward = float(mean_reward)
            current_session.completed_at = now
        else:
            # Create a new completed session if none exists
            current_session = TrainSession(
                model_id=int(model_id),
                user_id=int(user_id),
                timesteps=int(timesteps),
                total_time=float(total_time),
                mean_reward=float(mean_reward) if mean_reward is not None else None,
                started_at=now,
                completed_at=now
            )
            db.add(current_session)

        db.commit()
        return current_session
    except Exception as e:
        print(f"Error creating/completing training session: {str(e)}")
        db.rollback()
        return None
    finally:
        db.close()

def get_model_sessions(model_id: int):
    db = SessionLocal()
    try:
        return db.query(TrainSession).filter(TrainSession.model_id == model_id).all()
    finally:
        db.close()

def log_training(model_name: str, mean_reward: float, current_timestep: int, user_id: int):
    """Log training progress for a model by updating the current training session."""
    db = SessionLocal()
    try:
        # Find the model for this specific user
        model = (
            db.query(TrainedModel)
            .filter(TrainedModel.name == model_name)
            .filter(TrainedModel.user_id == user_id)
            .first()
        )
        
        if not model:
            print(f"Warning: Model {model_name} not found for user {user_id}")
            return

        # Find or create the current training session
        current_session = (
            db.query(TrainSession)
            .filter(TrainSession.model_id == model.id)
            .filter(TrainSession.user_id == user_id)  # Extra safety check
            .filter(TrainSession.completed_at.is_(None))  # Not completed yet
            .first()
        )

        if not current_session:
            # Create a new session if none exists
            current_session = TrainSession(
                model_id=model.id,
                user_id=user_id,
                timesteps=0,  # Will be updated when training completes
                total_time=0.0,  # Will be updated when training completes
                started_at=datetime.utcnow()
            )
            db.add(current_session)
            db.commit()  # Commit to get the session ID
            db.refresh(current_session)

        # Add a new progress log entry
        progress = TrainingProgress(
            session_id=current_session.id,
            timestep=current_timestep,
            mean_reward=mean_reward
        )
        db.add(progress)
        db.commit()
    except Exception as e:
        print(f"Error logging training progress: {str(e)}")
        db.rollback()
    finally:
        db.close()

def fetch_logs(model_name: str, user_id: int):
    """Fetch training logs for a model belonging to a specific user."""
    db = SessionLocal()
    try:
        # Find the model for this specific user
        model = (
            db.query(TrainedModel)
            .filter(TrainedModel.name == model_name)
            .filter(TrainedModel.user_id == user_id)
            .first()
        )
        
        if not model:
            print(f"Warning: Model {model_name} not found for user {user_id}")
            return []

        # Get all training sessions for this model
        sessions = (
            db.query(TrainSession)
            .filter(TrainSession.model_id == model.id)
            .filter(TrainSession.user_id == user_id)  # Extra safety check
            .order_by(TrainSession.started_at)
            .all()
        )
        
        # Format the logs
        logs = []
        for session in sessions:
            # Get all progress logs for this session
            progress_logs = (
                db.query(TrainingProgress)
                .filter(TrainingProgress.session_id == session.id)
                .order_by(TrainingProgress.timestep)
                .all()
            )
            
            session_data = {
                'session_id': session.id,
                'model_id': model.id,
                'user_id': user_id,
                'model_name': model_name,
                'started_at': session.started_at.isoformat() if session.started_at else None,
                'completed_at': session.completed_at.isoformat() if session.completed_at else None,
                'is_completed': session.completed_at is not None,
                'total_time': float(session.total_time) if session.total_time else 0.0,
                'final_timesteps': int(session.timesteps) if session.timesteps else 0,
                'progress': [
                    {
                        'timestep': int(log.timestep),
                        'mean_reward': float(log.mean_reward) if log.mean_reward is not None else None,
                        'logged_at': log.logged_at.isoformat() if log.logged_at else None
                    }
                    for log in progress_logs
                ]
            }
            logs.append(session_data)
        
        return logs
    finally:
        db.close()

if __name__ == "__main__":
    init_db()
    print("✅ Tables created successfully.")
