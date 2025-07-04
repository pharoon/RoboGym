import os
import shutil
from typing import Optional, Dict
from stable_baselines3 import PPO

from model_manager.manager import ModelManager
from rl_agent.train_agent import RLTrainer

TASK_CHOICES = {
    1: "pick_and_place",
    # 2: "button_pressing",
    # 3: "path_following",
}


class ModelService:
    def __init__(self):
        self.model_manager = ModelManager()
        self.trainer = RLTrainer()
    def initialize(self):
        print(" Initializing RoboGym environment...")
        os.makedirs("trained_models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        print(" Directories and database initialized.")

    def train(self, model_name: str, timesteps: int, task_number: int,
              learning_rate: float, batch_size: int, n_steps: int,
              model_path: Optional[str] = None):
        task_name = TASK_CHOICES.get(task_number)
        if not task_name:
            yield f"data: Invalid task number: {task_number}\n\n"
            return
       
        yield from self.trainer.train_model(
            total_timesteps=timesteps,
            model_name=model_name,
            task_name=task_name,
            model_path=model_path,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_steps=n_steps
        )

        yield "data:\n\n"
        yield "event: end\ndata: done\n\n"

    def test(self, model_name, model_path, task_name: str, episodes: int):
        RL_model = self.model_manager.load_model(model_name, model_path=model_path)
        yield f"data: Loaded model\n\n"
        yield from self.trainer.test_model(RL_model, task_name, episodes)

    def delete(self, model_name: str, model_path: Optional[str] = None):
        self.model_manager.delete_model(model_name, model_path)

    def upload_model(self, file_path: str, model_name: str, target_path: Optional[str] = None):
        if not os.path.exists(file_path):
            print(" File does not exist.")
            return

        dest_path = target_path or os.path.join("trained_models", f"{model_name}.zip")
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy(file_path, dest_path)

        model = PPO.load(dest_path)
        self.model_manager.save_model(model, model_name, model_path=dest_path)
        print(f" Model uploaded and registered as '{model_name}'")


    def rename_local_model_file(self, user_id: int, old_name: str, new_name: str) -> str:
        
        user_folder = f"trained_models/user_{user_id}"
        old_path = os.path.join(user_folder, f"{old_name}.zip")
        new_path = os.path.join(user_folder, f"{new_name}.zip")

       
        if not os.path.exists(old_path):
            return new_path

        try:
            os.rename(old_path, new_path)
        except Exception as e:
            print(f"Failed to rename file: {e}. Continuing without error.")

        return new_path



    