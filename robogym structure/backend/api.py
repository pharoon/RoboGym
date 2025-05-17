
from fastapi import FastAPI
from model_manager import manager
from rl_agent.train_agent import train_rl_model
import os

app = FastAPI()

@app.get("/models")
def get_models():
    return {"models": manager.list_models()}

@app.post("/train")
def train_model():
    os.makedirs("trained_models", exist_ok=True)
    train_rl_model()
    return {"status": "training complete"}
