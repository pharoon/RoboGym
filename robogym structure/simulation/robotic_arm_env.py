import pybullet as p
import pybullet_data
import gym
import numpy as np
import os
from gym import spaces

class RoboticArmEnv(gym.Env):
    """
    Custom Gym-compatible environment for simulating a robotic arm using PyBullet.
    Compatible with RL libraries (e.g., Stable-Baselines3).
    """

    def __init__(self, render: bool = False):
        super(RoboticArmEnv, self).__init__()

        # Start physics simulation
        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Load environment and robot
        self.robot_urdf = os.path.join(pybullet_data.getDataPath(), "kuka_iiwa/model.urdf")
        self._load_environment()

        # Set control parameters
        self.num_joints = p.getNumJoints(self.robot_id)
        self.max_force = 500

        # Define Gym-compatible action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.pi, high=np.pi, shape=(self.num_joints,), dtype=np.float32)

    def _load_environment(self):
        """Loads the plane and robotic arm into the simulation."""
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(self.robot_urdf, basePosition=[0, 0, 0], useFixedBase=True)

    def _get_observation(self):
        """Returns current joint angles as observation."""
        return np.array([p.getJointState(self.robot_id, i)[0] for i in range(self.num_joints)], dtype=np.float32)

    def step(self, action):
        """Applies action, steps the simulation, and returns observation, reward, done, and info."""
        action = np.clip(action, -1.0, 1.0)
        for i in range(self.num_joints):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=action[i],
                force=self.max_force
            )
        p.stepSimulation()

        obs = self._get_observation()
        reward = -np.sum(np.square(action))  # example: minimize movement
        done = False
        info = {}

        return obs, reward, done, info

    def reset(self):
        """Resets the environment to initial state."""
        self._load_environment()
        return self._get_observation()

    def render(self, mode="human"):
        """Optional custom render logic."""
        pass  # Not needed with PyBullet GUI

    def close(self):
        """Disconnects the simulation."""
        p.disconnect()
