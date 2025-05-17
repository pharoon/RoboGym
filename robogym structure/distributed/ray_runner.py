
import ray
from stable_baselines3 import PPO
from simulation.robotic_arm_env import RoboticArmEnv

ray.init(ignore_reinit_error=True)

@ray.remote
def train_distributed_agent():
    env = RoboticArmEnv()
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=5000)
    env.close()
    return "done"

futures = [train_distributed_agent.remote() for _ in range(4)]
results = ray.get(futures)
print(results)
