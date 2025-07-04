import enum

class AlgorithmType(enum.Enum):
    PPO = "PPO"
    DQN = "DQN"
    SAC = "SAC"
    TD3 = "TD3"

class RoboticArmType(enum.Enum):
    KUKA_IIWA = "kuka_iiwa"
    UR5 = "ur5"
    PANDA = "panda"
