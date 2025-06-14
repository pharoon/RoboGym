import numpy as np
import pybullet as p
from tasks.base_task import BaseTask

class PickAndPlaceTask(BaseTask):
    def __init__(self, env, *, switch_tables=False):
        super().__init__(env)
        self.switch_tables = switch_tables
        self.source_marker = None
        self.target_marker = None
        self.prev_dist = None



    def reset(self):
        TABLE_HEIGHT = 0.4
        OBJECT_HEIGHT = 0.05

        x_pos = 0.65
        y_offset = 0.3

        if not self.switch_tables:
            source_y = +y_offset
            target_y = -y_offset
        else:
            source_y = -y_offset
            target_y = +y_offset

        self.env.source_pos = np.array([
            x_pos + np.random.uniform(-0.05, 0.05),
            source_y + np.random.uniform(-0.05, 0.05),
            TABLE_HEIGHT + OBJECT_HEIGHT / 2
        ])

        self.env.target_pos = np.array([
            x_pos + np.random.uniform(-0.05, 0.05),
            target_y + np.random.uniform(-0.05, 0.05),
            TABLE_HEIGHT + OBJECT_HEIGHT / 2
        ])

        self.env.phase = "pick"
        self.env.touched_source = False

        if self.source_marker is None:
            self.source_marker = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=p.createVisualShape(
                    shapeType=p.GEOM_SPHERE,
                    radius=0.03,
                    rgbaColor=[1, 0, 0, 0.4]  # Red, 40% transparent
                ),
                basePosition=self.env.source_pos.tolist()
            )

        if self.target_marker is None:
            self.target_marker = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=p.createVisualShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=[0.03, 0.03, 0.03],
                    rgbaColor=[0, 1, 0, 0.4]  # Green, 40% transparent
                ),
                basePosition=self.env.target_pos.tolist()
            )

        p.resetBasePositionAndOrientation(self.source_marker, self.env.source_pos.tolist(), [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.target_marker, self.env.target_pos.tolist(), [0, 0, 0, 1])


        if hasattr(self.env, "object_id") and self.env.object_id is not None:
            p.resetBasePositionAndOrientation(self.env.object_id, self.env.source_pos.tolist(), [0, 0, 0, 1])
   
   
    def compute_reward(self, ee_pos):
        reward = 0.0
        done = False

        if self.env.phase == "pick":
            dist_to_source = np.linalg.norm(np.array(ee_pos) - self.env.source_pos)
            reward = -dist_to_source  

            if dist_to_source < 0.08:
                reward += 1.0
                self.env.phase = "place"
                self.env.touched_source = True

        elif self.env.phase == "place":
            
            dist_to_target = np.linalg.norm(np.array(ee_pos) - self.env.target_pos)
            if self.prev_dist is not None:
                reward += (self.prev_dist - dist_to_target) * 2.0  # Reward progress
            self.prev_dist = dist_to_target

            if dist_to_target < 0.08 and self.env.touched_source:
                reward += 3.0
                print("[DEBUG] Task completed successfully!")
                done = True

        reward *= 5.0  

        return reward, done, {
            "phase": self.env.phase,
            "touched_source": self.env.touched_source
        }
