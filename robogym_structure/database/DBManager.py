# models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, Enum, ForeignKey, create_engine, JSON,func
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timezone
from database.models import User, TrainedModel, TrainSession,UserStats
import os
from dotenv import load_dotenv
from database.FileStorage import FileManager
# Load environment variables
load_dotenv()   



SQLALCHEMY_DATABASE_URL = f"postgresql://postgres.xljntaujspiljiczjzzh:3aCXHe0fDQx0gzNL@aws-0-eu-central-1.pooler.supabase.com:6543/postgres"
# SQLAlchemy setup
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class DBManager:
    def __init__(self):
        self.db = SessionLocal()
        

    def close(self):
        self.db.close()

    # --- User Utilities ---
    def create_user(self, username: str, email: str, password: str):
        try:
            user = User(username=username, email=email)
            user.set_password(password)
            self.db.add(user)
            self.db.flush()
            stats = UserStats(user_id=user.id)
            self.db.add(stats)
            self.db.commit()
            self.db.refresh(user)
            return user
        finally:
            self.db.close()

    def get_user_by_username(self, username: str):
        return self.db.query(User).filter(User.username == username).first()

    def get_user_by_email(self, email: str):
        return self.db.query(User).filter(User.email == email).first()

    def get_user_by_id(self, user_id: int):
        return self.db.query(User).filter(User.id == user_id).first()

    # --- User Stats ---
    def get_user_stats(self, user_id: int):
        stats = self.db.query(UserStats).filter(UserStats.user_id == user_id).first()
        if not stats:
            print(f"[WARN] No stats found for user_id={user_id}")
            return None
        return {
            "tests_run": stats.tests_run,
            "train_sessions_count": stats.train_sessions_count,
            "trained_models_count": stats.trained_models_count
        }

    def _increment_counter(self, user_id: int, field: str):
        try:
            stats = self.db.query(UserStats).filter(UserStats.user_id == user_id).first()
            if stats:
                setattr(stats, field, getattr(stats, field) + 1)
                self.db.commit()
                return True
            print(f"[WARN] UserStats not found for user_id={user_id}")
            return False
        except Exception as e:
            print(f"[ERROR] increment_{field} failed: {e}")
            self.db.rollback()
            return False

    def increment_tests_run(self, user_id: int):
        return self._increment_counter(user_id, "tests_run")

    def increment_train_sessions_count(self, user_id: int):
        return self._increment_counter(user_id, "train_sessions_count")

    def increment_models_count(self, user_id: int):
        return self._increment_counter(user_id, "trained_models_count")

    def update_user_stats(self, user_id: int):
        try:
            stats = self.db.query(UserStats).filter(UserStats.user_id == user_id).first()
            if not stats:
                print(f"[WARN] No stats found for user_id={user_id}")
                return None
            stats.train_sessions_count += 1
            stats.trained_models_count += 1
            self.db.commit()
            return True
        except Exception as e:
            print(f"[ERROR] update_user_stats failed: {e}")
            self.db.rollback()
            return False

    # --- Trained Models ---
    def create_trained_model(self, name, model_path, algorithm, robotic_arm, user_id, timesteps=0, total_time=0.0, mean_reward=None):
        try:
            model = TrainedModel(
                name=name,
                model_path=model_path,
                algorithm=algorithm,
                robotic_arm=robotic_arm,
                user_id=user_id,
                total_timesteps=timesteps,
                total_training_time=total_time,
                final_mean_reward=mean_reward
            )
            self.db.add(model)
            stats = self.db.query(UserStats).filter(UserStats.user_id == user_id).first()
            if stats:
                stats.trained_models_count += 1
            self.db.commit()
            self.db.refresh(model)
            return model
        except Exception as e:
            print(f"Error creating training model: {e}")
            self.db.rollback()
            return None

    def get_user_models(self, user_id: int):
        return self.db.query(TrainedModel).filter(TrainedModel.user_id == user_id).all()

    def get_trained_model_by_name_and_user(self, user_id: int, model_name: str):
        return self.db.query(TrainedModel).filter_by(name=model_name, user_id=user_id).first()

    def delete_trained_model(self, user_id: int, model_name: str):
        try:
            model = self.get_trained_model_by_name_and_user(user_id, model_name)
            if not model:
                print(f"Model '{model_name}' not found for user ID {user_id}")
                return False
            self.db.delete(model)
            stats = self.db.query(UserStats).filter(UserStats.user_id == user_id).first()
            if stats:
                stats.trained_models_count -= 1
            self.db.commit()
            return True
        except Exception as e:
            print(f"Error deleting model: {e}")
            self.db.rollback()
            return False

    def update_trained_model(self, model_id: int, timesteps: int, total_time: float, mean_reward: float, new_model_path: str):
        try:
            model = self.db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
            if not model:
                print(f"Model with ID {model_id} not found.")
                return False
            model.model_path = new_model_path
            model.total_timesteps += timesteps
            model.total_training_time += total_time
            model.final_mean_reward = mean_reward
            self.db.commit()
            return True
        except Exception as e:
            print(f"Error updating model: {e}")
            self.db.rollback()
            return False

    def model_rename(self, user_id: int, old_name: str, new_name: str):
        try:
            if not new_name:
                print("New model name cannot be empty.")
                return False
            model = self.get_trained_model_by_name_and_user(user_id, old_name)
            if not model:
                print(f"Model '{old_name}' not found for user ID {user_id}")
                return False
            existing_model = self.get_trained_model_by_name_and_user(user_id, new_name)
            if existing_model:
                print(f"Model '{new_name}' already exists for user ID {user_id}")
                return False
            bucket = "models"
            old_key = extract_storage_key(model.model_path, bucket)
            new_key = old_key.replace(old_name, new_name)
            FM=FileManager()
            success = FM.rename_file(bucket, old_key, new_key)
            if not success:
                return False
            model.name = new_name
            model.model_path = model.model_path.replace(old_name, new_name)  
            self.db.commit()
            return True
        except Exception as e:
            print(f"Error renaming model: {e}")
            self.db.rollback()
            return False

    # --- TrainSession ---
    def create_train_session(self, model_id, user_id, timesteps, total_time, mean_reward=None, train_log=None):
        try:
            now = datetime.now(timezone.utc)
            session = TrainSession(
                model_id=model_id,
                timesteps=timesteps,
                total_time=total_time,
                mean_reward=mean_reward,
                started_at=now,
                completed_at=now,
                train_log=train_log
            )
            self.db.add(session)
            stats = self.db.query(UserStats).filter(UserStats.user_id == user_id).first()
            if stats:
                stats.train_sessions_count += 1
            self.db.commit()
            self.db.refresh(session)
            return session
        except Exception as e:
            print(f"Error creating training session: {e}")
            self.db.rollback()
            return None

    def get_model_sessions(self, model_id: int):
        return self.db.query(TrainSession).filter(TrainSession.model_id == model_id).all()

    def fetch_logs(self, model_name: str, user_id: int):
        model = self.get_trained_model_by_name_and_user(user_id, model_name)
        if not model:
            print(f"Model {model_name} not found for user {user_id}")
            return []

        sessions = self.get_model_sessions(model.id)
        logs = []
        for s in sessions:
            progress = []
            if s.train_log:
                try:
                    import json
                    train_log_data = json.loads(s.train_log)
                    if isinstance(train_log_data, list):
                        progress = train_log_data
                except Exception:
                    if s.mean_reward is not None:
                        progress = [{
                            'timestep': s.timesteps,
                            'mean_reward': s.mean_reward,
                            'logged_at': s.completed_at.isoformat() if s.completed_at else None
                        }]

            logs.append({
                'session_id': s.id,
                'model_id': model.id,
                'user_id': user_id,
                'model_name': model_name,
                'started_at': s.started_at.isoformat() if s.started_at else None,
                'completed_at': s.completed_at.isoformat() if s.completed_at else None,
                'is_completed': s.completed_at is not None,
                'total_time': float(s.total_time) if s.total_time else 0.0,
                'final_timesteps': int(s.timesteps) if s.timesteps else 0,
                'progress': progress
            })
        return logs

def extract_storage_key(public_url: str, bucket_name: str) -> str:
    """
    Extracts the storage key from a public Supabase URL.

    Example:
        public_url: https://abc.supabase.co/storage/v1/object/public/models/user_10/old_model.zip
        bucket_name: "models"
        → returns: "user_10/old_model.zip"
    """
    prefix = f"/storage/v1/object/public/{bucket_name}/"
    if prefix in public_url:
        return public_url.split(prefix, 1)[-1]
    raise ValueError("Invalid Supabase public URL format.")
