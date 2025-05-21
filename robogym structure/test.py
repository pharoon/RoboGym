import time
import pybullet as p
import pybullet_data
from simulation.robotic_arm_env import RoboticArmEnv  # Adjust import path if needed
from tasks.pick_and_place import PickAndPlaceTask  # Adjust import path if needed
from stable_baselines3 import PPO  # or TD3, SAC depending on what you trained with
def test_model(model_path, max_steps=100000):
    # Do NOT call p.connect() here! Let RoboticArmEnv handle it
    env = RoboticArmEnv(render=True)
    task = PickAndPlaceTask(env)
    env.task = task

    obs = env.reset()

    model = PPO.load(model_path)

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        print(f"Step {step} | Reward: {reward:.3f} | Phase: {info.get('phase', 'N/A')}")
        time.sleep(1.0 / 60.0)

        if done:
            print("✔️ Task completed or max steps reached.")
            break

    env.close()  # Proper cleanup


if __name__ == "__main__":
    # Path to your trained SB3 model
    model_path = "trained_models/test1.zip"  # Adjust as needed
    test_model(model_path)
