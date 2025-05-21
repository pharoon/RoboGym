import numpy as np
import pybullet as p
from tasks.base_task import BaseTask

class PickAndPlaceTask(BaseTask):
    def __init__(self, env):
        super().__init__(env)
        self.source_marker = None
        self.target_marker = None

    def reset(self):
        # Randomized pick and place positions (optional)
        self.env.source_pos = np.array([0.3 + np.random.uniform(0.0, 0.2), 0.2, 0.1])
        self.env.target_pos = np.array([0.3 + np.random.uniform(0.0, 0.2), -0.2, 0.1])
        self.env.phase = "pick"
        self.env.touched_source = False

        # Load or update markers
        if self.source_marker is None or self.target_marker is None:
            self.source_marker = p.loadURDF("sphere_small.urdf", [0, 0, 0], globalScaling=0.3)
            self.target_marker = p.loadURDF("cube_small.urdf", [0, 0, 0], globalScaling=0.3)

        p.resetBasePositionAndOrientation(self.source_marker, self.env.source_pos.tolist(), [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.target_marker, self.env.target_pos.tolist(), [0, 0, 0, 1])

    def compute_reward(self, ee_pos):
        reward = 0.0
        done = False

        if self.env.phase == "pick":
            dist = np.linalg.norm(np.array(ee_pos) - self.env.source_pos)
            reward = 1.0 - dist  # closer is better
            if dist < 0.05:
                self.env.phase = "place"
                self.env.touched_source = True
                reward += 0.5  # small bonus for touching source

        elif self.env.phase == "place":
            dist = np.linalg.norm(np.array(ee_pos) - self.env.target_pos)
            reward = 1.0 - dist
            if dist < 0.05 and self.env.touched_source:
                reward += 1.5  # big bonus for completing place
                done = True

        return reward, done, {
            "phase": self.env.phase,
            "touched_source": self.env.touched_source
        }
