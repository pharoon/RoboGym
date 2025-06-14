import pybullet as p
import pybullet_data
import gym
import numpy as np
import os
from gym import spaces

class RoboticArmEnv(gym.Env):
    def __init__(self, task=None, render: bool = False):
        super(RoboticArmEnv, self).__init__()
        self.task = task
        self.object_id = None  # Simulated grasped object

        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self.robot_urdf = os.path.join(pybullet_data.getDataPath(), "kuka_iiwa/model.urdf")
        self._load_environment()
        self.num_joints = p.getNumJoints(self.robot_id)
        self.max_force = 500

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32)
        obs_dim = self.num_joints + 3 + 3 + 3  # joint_angles + ee_pos + obj_pos + target_pos
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )      
        self.max_steps = 300
        self.current_step = 0
        self.total_reward = 0.0

        

    def _load_environment(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
    
        plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(self.robot_urdf, basePosition=[0, 0, 0], useFixedBase=True)

        # Load grabbable object
        if self.object_id is None:
            self.object_id = p.loadURDF("cube_small.urdf", [0, 0, 0], globalScaling=1.0)
        


        TABLE_HALF_EXTENTS = [0.2, 0.2, 0.2]  # = 40cm x 40cm x 40cm tall table
        TABLE_HEIGHT = TABLE_HALF_EXTENTS[2] * 2  # Full height = 0.4m
        OBJECT_HEIGHT = 0.05  # Adjust based on your object's real size

        # Pick table (Y+)
        self.table_source_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=TABLE_HALF_EXTENTS
            ),
            baseVisualShapeIndex=p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=TABLE_HALF_EXTENTS,
                rgbaColor=[0.6, 0.3, 0.1, 1]
            ),
            basePosition=[0.65, 0.3, TABLE_HALF_EXTENTS[2]],  # X = 0.5 to move forward
            useMaximalCoordinates=True
        )

        # Place table (Y-)
        self.table_target_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=TABLE_HALF_EXTENTS
            ),
            baseVisualShapeIndex=p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=TABLE_HALF_EXTENTS,
                rgbaColor=[0.2, 0.4, 0.7, 1]
            ),
            basePosition=[0.65, -0.3, TABLE_HALF_EXTENTS[2]],
            useMaximalCoordinates=True
        )

        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=50,
            cameraPitch=-35,
            cameraTargetPosition=[0, 0, 0]
        )

    def _get_observation(self):
        joint_angles = [p.getJointState(self.robot_id, i)[0] for i in range(self.num_joints)]
        ee_pos = p.getLinkState(self.robot_id, self.num_joints - 1)[0]  # End-effector position
        obj_pos = p.getBasePositionAndOrientation(self.object_id)[0]  # Object position
        if hasattr(self, "phase") and self.phase == "pick":
            current_target = self.source_pos
        else:
            current_target = self.target_pos

        
        obs = np.concatenate([
            joint_angles,          # self.num_joints (e.g. 7 for KUKA)
            ee_pos,                # 3 values
            obj_pos,               # 3 values
            current_target          # 3 values
        ])
        return np.array(obs, dtype=np.float32)

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

        # Simulate holding object
        if hasattr(self, "touched_source") and self.touched_source:
            if self.object_id is not None:
                p.resetBasePositionAndOrientation(self.object_id, ee_pos, [0, 0, 0, 1])

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

        if self.object_id:
            p.resetBasePositionAndOrientation(self.object_id, self.source_pos.tolist(), [0, 0, 0, 1])

        return self._get_observation()

    def render(self, mode="human"):
        pass

    def close(self):
        p.disconnect()
