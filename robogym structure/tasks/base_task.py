class BaseTask:
    def __init__(self, env):
        self.env = env

    def reset(self):
        """Optional task-specific reset."""
        pass

    def compute_reward(self, ee_pos):
        """
        Should return: reward, done (bool), info (dict or None)
        """
        raise NotImplementedError
