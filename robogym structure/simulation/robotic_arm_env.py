
import pybullet as p
import pybullet_data
import gym
import numpy as np
import os
from gym import spaces

class RoboticArmEnv(gym.Env):
    def __init__(self, render=False):
        super(RoboticArmEnv, self).__init__()
        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.robot_urdf = os.path.join(pybullet_data.getDataPath(), "kuka_iiwa/model.urdf")
        self._load_environment()
        self.num_joints = p.getNumJoints(self.robot_id)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-3.14, high=3.14, shape=(self.num_joints,), dtype=np.float32)

    def _load_environment(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(self.robot_urdf, useFixedBase=True)

    def step(self, action):
        for i in range(self.num_joints):
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, targetPosition=action[i], force=500)
        p.stepSimulation()
        obs = self._get_observation()
        reward = -np.linalg.norm(action)
        done = False
        return obs, reward, done, {}

    def reset(self):
        self._load_environment()
        return self._get_observation()

    def _get_observation(self):
        return np.array([p.getJointState(self.robot_id, i)[0] for i in range(self.num_joints)], dtype=np.float32)

    def close(self):
        p.disconnect()
