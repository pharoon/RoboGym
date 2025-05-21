import pybullet as p
import pybullet_data
import gym
import numpy as np
import os
from gym import spaces

class RoboticArmEnv(gym.Env):
    def __init__(self,task=None, render: bool = False):
        super(RoboticArmEnv, self).__init__()
        self.task = task
        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self.robot_urdf = os.path.join(pybullet_data.getDataPath(), "kuka_iiwa/model.urdf")
        self._load_environment()
        print("Robot IDw:", self.robot_id)  # 👈 MUST NOT be -1
        self.num_joints = p.getNumJoints(self.robot_id)
        self.max_force = 500

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.pi, high=np.pi, shape=(self.num_joints,), dtype=np.float32)

        # Track episode stats
        self.max_steps = 50
        self.current_step = 0
        self.total_reward = 0.0

    def _load_environment(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
    
        plane_id = p.loadURDF("plane.urdf")
        print("Plane ID:", plane_id)  # Should NOT be -1
    
        self.robot_id = p.loadURDF(self.robot_urdf, basePosition=[0, 0, 0], useFixedBase=True)
        print("Robot ID:", self.robot_id)  # Should be >= 0
    
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=50,
            cameraPitch=-35,
            cameraTargetPosition=[0, 0, 0]
        )

        
    def _get_observation(self):
        return np.array([p.getJointState(self.robot_id, i)[0] for i in range(self.num_joints)], dtype=np.float32)

    def step(self, action):
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
        ee_pos = p.getLinkState(self.robot_id, self.num_joints - 1)[0]

        if self.task:
           
            reward, task_done, info = self.task.compute_reward(ee_pos)
        else:
            reward = -np.sum(np.square(action))
            task_done = False
            info = {}

        self.total_reward += reward
        self.current_step += 1
        done = task_done or self.current_step >= self.max_steps

        if done:
            info["episode"] = {"r": self.total_reward, "l": self.current_step}  

        return obs, reward, done, info

    def reset(self):
        self.current_step = 0
        self.total_reward = 0.0
        for i in range(self.num_joints):
            p.resetJointState(self.robot_id, i, targetValue=0.0)
        if self.task:
            self.task.reset()
        return self._get_observation()

    def render(self, mode="human"):
        pass

    def close(self):
        p.disconnect()
