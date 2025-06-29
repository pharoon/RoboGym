import os
import argparse

from model_manager import manager as mm
from rl_agent.train_agent import train_model, test_model
from database import models as db
from analytics import logger

TASK_CHOICES = {
    1: "pick_and_place",
    # 2: "button_pressing",
    # 3: "path_following",
}

def initialize():
    print(" Initializing RoboGym environment...")
    
    os.makedirs("trained_models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    print(" Directories and database initialized.")

def train(model_name, timesteps, task_number, model_path=None):
    task_name = TASK_CHOICES[task_number]
    yield from train_model(timesteps, model_name, task_name, model_path=model_path)
    yield "data:\n\n"  # <-- this dummy message helps flush the stream
    yield "event: end\ndata: done\n\n"

def test(RL_model, task_name, episodes):
    yield from test_model(RL_model, task_name, episodes)

def list_models():
    models = mm.list_models()
    if not models:
        print(" No models found.")
    for m in models:
        print(f" {m['name']} | Created: {m['created_at']} | Algorithm: {m['algorithm']}")

def delete(model_name, model_path=None):
    mm.delete_model(model_name, model_path=model_path)

def analytics_menu():
    print("1. Plot rewards for a model")
    print("2. Compare multiple models")
    choice = input("Choice: ")
    if choice == "1":
        model_name = input("Model name: ")
        logger.plot_model_rewards(model_name)
    elif choice == "2":
        model_names = input("Enter model names separated by commas: ").split(",")
        model_names = [m.strip() for m in model_names]
        logger.compare_models(model_names)
    else:
        print("Invalid choice.")

def upload_model(file_path, model_name, target_path=None):
    from stable_baselines3 import PPO
    import shutil

    if not os.path.exists(file_path):
        print(" File does not exist.")
        return
        
    if target_path:
        dest_path = target_path
    else:
        dest_path = os.path.join("trained_models", f"{model_name}.zip")
        
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy(file_path, dest_path)

    model = PPO.load(dest_path)
    mm.save_model(model, model_name, model_path=dest_path)
    print(f" Model uploaded and registered as '{model_name}'")

def test_model_stream(model_name: str, task_number: int, episodes: int, model_path=None):
    RL_model = mm.load_model(model_name, model_path=model_path)
    task_name = TASK_CHOICES.get(task_number)

    if not task_name:
        yield "❌ Invalid task number."
        return

    yield f"🧪 Testing model '{model_name}' on task '{task_name}' for {episodes} episodes...\n"
    try:
        for line in test_model(RL_model, task_name, episodes, stream=True):  
            yield line
    except Exception as e:
        yield f"🚨 Error during testing: {str(e)}"

def GetModelRewards(modelName):
    logger.plot_model_rewards(model_name=modelName)

def compareModels(model1, model2):
    logger.compare_models([model1,model2])