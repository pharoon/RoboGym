import os
from stable_baselines3 import PPO
from simulation.robotic_arm_env import RoboticArmEnv
from stable_baselines3.common.callbacks import BaseCallback
from model_manager import manager as mm  
from database import models as db 
from tasks.pick_and_place import PickAndPlaceTask
import time

TASK_MAP = {
    "pick_and_place": PickAndPlaceTask,
    # Add more tasks here
}

MODELS_DIR = "trained_models"
LOGS_DIR = "logs"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Custom callback to save the best model based on training reward.
    """
    def __init__(self, check_freq: int, save_path: str,model_name: str, verbose=1, ):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -float("inf")
        self.model_name = model_name

    def _on_step(self) -> bool:

        if self.n_calls % self.check_freq == 0:
            
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = sum([ep["r"] for ep in self.model.ep_info_buffer]) / len(self.model.ep_info_buffer)
                db.log_training(self.model_name, mean_reward)

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    print("New best reward! Saving model...")
                    self.model.save(self.save_path)
        return True


def train_model(total_timesteps=10000, model_name="ppo_robotic_arm",task_name="pick_and_place"):
    task_class = TASK_MAP.get(task_name)
    if not task_class:
        raise ValueError(f"Task '{task_name}' not found.")
    from simulation.robotic_arm_env import RoboticArmEnv
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    import os

    dummy_env = RoboticArmEnv()  # just to pass to the task class
    task_instance = task_class(dummy_env)
    env = RoboticArmEnv(task=task_instance, render=False)

    task_instance.env = env  # link the real env
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOGS_DIR)

    save_callback = SaveOnBestTrainingRewardCallback(
        check_freq=10,
        save_path=os.path.join(MODELS_DIR, model_name),
        model_name=model_name
    )

    model.learn(total_timesteps=total_timesteps, callback=save_callback)

    # Save final model
    final_model_path = os.path.join(MODELS_DIR, f"{model_name}.zip")
    model.save(final_model_path)
    print("finished")
    # ✅ Register model in metadata
    mm.save_model(model, model_name)

    print(f"[✓] Final model saved to: {final_model_path}")
    env.close()
    return final_model_path

def test_model(model, task_name: str, episodes: int = 5):
    

    # Lookup task
    if task_name not in TASK_MAP:
        raise ValueError(f"Unknown task: {task_name}")
    
    # Setup task and environment
    dummy_env = RoboticArmEnv()
    task_instance = TASK_MAP[task_name](dummy_env)
    env = RoboticArmEnv(task=task_instance, render=True)
    task_instance.env = env  
    time.sleep(1)
    total_rewards = []

    for episode in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        print(f"\n🎬 Episode {episode + 1} started...")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            time.sleep(1. / 240.)  # smooth visualization

        print(f"[✓] Episode {episode + 1} complete — Total Reward: {ep_reward:.2f}")
        total_rewards.append(ep_reward)

    env.close()

    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"\n📊 Average Reward over {episodes} episodes: {avg_reward:.2f}")    