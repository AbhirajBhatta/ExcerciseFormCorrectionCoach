import random
import numpy as np

class PushupSimulator:

    def __init__(self):
        self.IDEAL = {
            "elbow_angle": 90,
            "back_angle": 180,
            "hip_angle": 180
        }
        self.reset()

    def reset(self):
        self.steps=0
        self.state = {
            "elbow_angle": random.uniform(100, 160),
            "back_angle": random.uniform(130, 170),
            "hip_angle": random.uniform(130, 170)
        }
        return self.get_state_vector()

    def get_state_vector(self):
        return np.array([
            self.state["elbow_angle"],
            self.state["back_angle"],
            self.state["hip_angle"]
        ])

    def compute_error(self):
        error = 0
        for key in self.state:
            error += abs(self.state[key] - self.IDEAL[key])
        return error

    def add_noise(self):
        for key in self.state:
            self.state[key] += random.uniform(-1, 1)

    def step(self, action):
        self.steps += 1

        prev_error = self.compute_error()

        if action == 0:
            self.state["elbow_angle"] -= random.uniform(2, 5)
        elif action == 1:
            self.state["back_angle"] += random.uniform(2, 5)
        elif action == 2:
            self.state["hip_angle"] += random.uniform(2, 5)

        self.add_noise()

        self.state["elbow_angle"] = max(60, min(180, self.state["elbow_angle"]))
        self.state["back_angle"] = max(120, min(200, self.state["back_angle"]))
        self.state["hip_angle"] = max(120, min(200, self.state["hip_angle"]))

        new_error = self.compute_error()

        reward = (prev_error - new_error) / 10

        # ✅ FIXED DONE
        done = self.steps >= 30

        return self.get_state_vector(), reward, done