
import os
from stable_baselines3 import PPO

MODEL_DIR = "trained_models"

def list_models():
    return [f for f in os.listdir(MODEL_DIR) if f.endswith('.zip')]

def load_model(model_name):
    path = os.path.join(MODEL_DIR, model_name)
    return PPO.load(path)

def save_model(model, model_name):
    path = os.path.join(MODEL_DIR, model_name)
    model.save(path)
