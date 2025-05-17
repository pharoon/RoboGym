import os
from stable_baselines3 import PPO
from simulation.robotic_arm_env import RoboticArmEnv
from stable_baselines3.common.callbacks import BaseCallback
import datetime

MODELS_DIR = "trained_models"
LOGS_DIR = "logs"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Custom callback to save the best model based on training reward.
    """
    def __init__(self, check_freq: int, save_path: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -float("inf")

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = sum([ep["r"] for ep in self.model.ep_info_buffer]) / len(self.model.ep_info_buffer)
                if self.verbose > 0:
                    print(f"Step {self.n_calls}, Mean reward: {mean_reward:.2f}")
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    print("New best reward! Saving model...")
                    self.model.save(self.save_path)
        return True


def train_model(total_timesteps=10000, model_name="ppo_robotic_arm"):
    """
    Trains a PPO agent on the RoboticArmEnv and saves it.
    """
    env = RoboticArmEnv(render=False)

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOGS_DIR)

    save_callback = SaveOnBestTrainingRewardCallback(
        check_freq=1000,
        save_path=os.path.join(MODELS_DIR, model_name)
    )

    model.learn(total_timesteps=total_timesteps, callback=save_callback)

    final_model_path = os.path.join(MODELS_DIR, f"{model_name}_final")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    env.close()
    return final_model_path


def test_model(model_path, episodes=5):
    """
    Loads a trained model and runs it for a few episodes in GUI mode.
    """
    env = RoboticArmEnv(render=True)
    model = PPO.load(model_path)

    for episode in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

    env.close()
