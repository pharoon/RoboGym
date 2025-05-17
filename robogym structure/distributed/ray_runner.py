import ray
from stable_baselines3 import PPO
from simulation.robotic_arm_env import RoboticArmEnv
from model_manager import manager
from database.models import log_training, init_db
import os

# Initialize Ray
ray.init(ignore_reinit_error=True, include_dashboard=False)

# Set up model saving directory
MODEL_DIR = "trained_models"
os.makedirs(MODEL_DIR, exist_ok=True)

@ray.remote
def train_model_parallel(model_name: str, total_timesteps: int = 5000):
    """
    Distributed task to train a PPO agent and return result.
    """
    env = RoboticArmEnv(render=False)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=total_timesteps)

    model_path = f"{MODEL_DIR}/{model_name}.zip"
    model.save(model_path)

    manager.save_model(model, model_name)
    log_training(model_name, -1)  # Placeholder reward (can be updated later)

    env.close()
    return f"[✓] Finished training: {model_name}"

def run_parallel_training(model_names, timesteps_per_model):
    """
    Accepts a list of model names and timesteps and trains them in parallel.
    """
    tasks = [
        train_model_parallel.remote(name, timesteps_per_model)
        for name in model_names
    ]
    results = ray.get(tasks)
    return results


if __name__ == "__main__":
    init_db()
    models = ["arm_agent_A", "arm_agent_B", "arm_agent_C"]
    timesteps = 8000

    print("⏳ Starting parallel training tasks...")
    result_logs = run_parallel_training(models, timesteps)

    for result in result_logs:
        print(result)

    ray.shutdown()
