import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.simulation.pushup_simulator import PushupSimulator

class PushupEnv(gym.Env):

    def __init__(self):
        super(PushupEnv, self).__init__()

        self.sim = PushupSimulator()

        # 3 actions
        self.action_space = spaces.Discrete(3)

        # 3 features
        self.observation_space = spaces.Box(
            low=np.array([60, 120, 120]),
            high=np.array([180, 200, 200]),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)   # important line
        state = self.sim.reset()
        return state, {}

    def step(self, action):
        state, reward, done = self.sim.step(action)
        return state, reward, done, False, {}