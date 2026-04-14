import gymnasium as gym
from gymnasium import spaces
import numpy as np
from config import FEATURE_NAMES


class GymPushupEnv(gym.Env):

    def __init__(self, env):
        super().__init__()

        self.env = env

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(FEATURE_NAMES),),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(4)

    def reset(self, seed=None, options=None):
        state = self.env.reset()
        return np.array(state, dtype=np.float32), {}

    def step(self, action):
        state, reward, done, _ = self.env.step(action)

        return (
            np.array(state, dtype=np.float32),
            reward,
            done,
            False,
            {}
        )