from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import model_manager.manager as mm
from rl_agent.train_agent import train_model, test_model
import os

app = FastAPI(title="RoboGym API")

# -----------------------------
# Models for request validation
# -----------------------------
class TrainRequest(BaseModel):
    model_name: str
    timesteps: int = 10000

class TestRequest(BaseModel):
    model_name: str
    episodes: int = 5

class DeleteRequest(BaseModel):
    model_name: str


# -----------------------------
# Routes
# -----------------------------

@app.get("/")
def index():
    return {"message": "RoboGym backend is running."}

@app.get("/models")
def get_models():
    return mm.list_models()

@app.post("/train")
def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(train_model, req.timesteps, req.model_name)
    return {"status": "training started", "model": req.model_name}

@app.post("/test")
def test_trained_model(req: TestRequest, background_tasks: BackgroundTasks):
    model_metadata = {m['name']: m for m in mm.list_models()}
    if req.model_name not in model_metadata:
        raise HTTPException(status_code=404, detail="Model not found.")
    model_path = model_metadata[req.model_name]["path"]
    background_tasks.add_task(test_model, model_path, req.episodes)
    return {"status": "test started", "model": req.model_name}

@app.delete("/model")
def delete_model(req: DeleteRequest):
    mm.delete_model(req.model_name)
    return {"status": f"model '{req.model_name}' deleted"}
