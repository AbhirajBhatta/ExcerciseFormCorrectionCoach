import numpy as np
import pandas as pd
import random

from simulator import PushupSimulator
from utils import dict_to_vector, vector_to_dict
from config import FEATURE_NAMES


class PushupEnv:

    def __init__(self, model, dataset_df):
        self.model = model
        self.dataset = dataset_df

        self.simulator = PushupSimulator()
        self.max_steps = 10

    def reset(self):
        bad_samples = self.dataset[self.dataset["label"] == 0]

        row = bad_samples.sample(1).iloc[0]

        state_dict = {f: row[f] for f in FEATURE_NAMES}
        self.state = dict_to_vector(state_dict)

        self.steps = 0
        return self.state

    def step(self, action):
        prev_score = self.get_score(self.state)

        next_state = self.simulator.step(self.state, action)

        curr_score = self.get_score(next_state)
# 
        reward = (curr_score - prev_score) * 2.5
        state_dict = vector_to_dict(next_state)

        prev_dict = vector_to_dict(self.state)
        curr_dict = vector_to_dict(next_state)

        reward += 0.5 * (curr_dict["max_depth"] - prev_dict["max_depth"])
        reward += 0.2 * (curr_dict["avg_alignment"] - prev_dict["avg_alignment"])

        if abs(curr_score - prev_score) < 1e-4:
            reward -= 0.005

# 

        reward = np.clip(reward, -0.2, 0.2)


        self.state = next_state
        self.steps += 1

        done = self.steps >= self.max_steps or (curr_score > 0.95 and self.steps > 3)

        return next_state, reward, done, {}

    def get_score(self, state_vec):
        X = pd.DataFrame([state_vec], columns=FEATURE_NAMES)
        prob = self.model.predict_proba(X)[0][1]
        return float(np.clip(prob, 0, 1))