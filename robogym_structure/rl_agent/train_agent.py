import os
from stable_baselines3 import PPO
from simulation.robotic_arm_env import RoboticArmEnv
from stable_baselines3.common.callbacks import BaseCallback
from model_manager.manager import ModelManager as mm

from tasks.pick_and_place import PickAndPlaceTask
import time
from typing import Optional
import io
import sys
import queue
import threading

TASK_MAP = {
    "pick_and_place": PickAndPlaceTask,
}

MODELS_DIR = "trained_models"
LOGS_DIR = "logs"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Custom callback to save the best model based on training reward and send reward updates in real time.
    """
    def __init__(self, check_freq: int, save_path: str, model_name: str, message_queue=None, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -float("inf")
        self.model_name = model_name
        self.message_queue = message_queue 

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = sum([ep["r"] for ep in self.model.ep_info_buffer]) / len(self.model.ep_info_buffer)
                if self.message_queue:
                    try:
                        self.message_queue.put_nowait(f"data: @timeStep: {self.num_timesteps} -> meanReward: {mean_reward}\n\n")

                    except queue.Full:
                        pass  
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.message_queue:
                        try:
                            self.message_queue.put_nowait(f"data: New best reward! Saving model...\n\n")
                        except queue.Full:
                            pass
                    self.model.save(self.save_path)
        return True

class RLTrainer:
    def __init__(self):
        self.models_dir = MODELS_DIR
        self.logs_dir = LOGS_DIR

    def train_model(self,total_timesteps=10000, model_name="ppo_robotic_arm", task_name="pick_and_place", model_path=None, learning_rate=1e-4, n_steps=2048, batch_size=64):
        yield(f"data: Starting training for model={model_name} on task={task_name} for {total_timesteps} timesteps\n\n")

        task_class = TASK_MAP.get(task_name)
        if not task_class:
            yield(f"data: Task '{task_name}' not found.\n\n")
            raise ValueError(f"Task '{task_name}' not found.")

        dummy_env = RoboticArmEnv()  
        task_instance = task_class(dummy_env)
        env = RoboticArmEnv(task=task_instance, render=False)

        task_instance.env = env  
        
        if model_path is None:
            model_path = os.path.join(self.models_dir, f"{model_name}.zip")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        if os.path.exists(model_path):
            yield(f"data: Found existing model at {model_path}. Loading and continuing training...\n\n")
            model = PPO.load(model_path, env=env, tensorboard_log=self.logs_dir)
        else:
            yield(f"data: No existing model found. Starting fresh training...\n\n")
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=self.logs_dir,
                learning_rate=float(learning_rate),     
                n_steps=int(n_steps),
                batch_size=int(batch_size),
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5
            )

        message_queue = queue.Queue(maxsize=100)  
        
        save_callback = SaveOnBestTrainingRewardCallback(
            check_freq=10,
            save_path=os.path.splitext(model_path)[0],  
            model_name=model_name,
            message_queue=message_queue
        )

        original_stdout = sys.stdout
        sys.stdout = io.StringIO()  

        training_done = threading.Event()
        
        def training_thread():
            try:
                model.learn(total_timesteps=total_timesteps, callback=save_callback)
            finally:
                training_done.set()

        train_thread = threading.Thread(target=training_thread)
        train_thread.start()

        while not training_done.is_set() or not message_queue.empty():
            try:
                message = message_queue.get(timeout=0.1)
                yield message
            except queue.Empty:
                continue

        train_thread.join()
        
        output = sys.stdout.getvalue()
        sys.stdout = original_stdout

        for line in output.splitlines():
            if line.strip():  
                yield f"data: {line}\n\n"

        model.save(model_path)
        mmm=mm()
        mmm.save_model(model, model_name, model_path=model_path)

        yield(f"data: Training complete. Model saved at {model_path}\n\n")
        env.close()
        return model_path

    def test_model(self,model, task_name: str, episodes: int = 5, stream: bool = False):
        if task_name not in TASK_MAP:
            yield(f"data:Unknown task: {task_name}\n\n")
            return

        yield(f"data: Setting up environment for task '{task_name}'\n\n")

        env = RoboticArmEnv(render=True)
        task = PickAndPlaceTask(env)
        env.task = task
        time.sleep(1)

        total_rewards = []

        for episode in range(episodes):
            obs = env.reset()
            done = False
            ep_reward = 0
            yield(f"data: Starting episode {episode + 1}/{episodes} with task switch={task.switch_tables}\n\n")
            yield(f"data: Episode {episode + 1} started...\n\n")

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                ep_reward += reward
                time.sleep(1. / 30.)  

            yield(f"data: Episode {episode + 1} complete — Total Reward: {ep_reward:.2f}\n\n")
            total_rewards.append(ep_reward)

        env.close()
        avg_reward = sum(total_rewards) / len(total_rewards)
        yield(f"data: Average Reward over {episodes} episodes: {avg_reward:.2f}\n\n")

