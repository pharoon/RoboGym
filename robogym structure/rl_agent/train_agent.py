
from stable_baselines3 import PPO
from simulation.robotic_arm_env import RoboticArmEnv

def train_rl_model(total_timesteps=10000, save_path='trained_models/ppo_robotic_arm'):
    env = RoboticArmEnv(render=False)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    env.close()
    print(f"Model saved to {save_path}")
