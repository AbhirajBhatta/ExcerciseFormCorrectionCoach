import numpy as np
from utils import vector_to_dict, dict_to_vector

class PushupSimulator:

    def __init__(self):
        self.success_prob = 0.85
    def clip_state(self, state):
        state["min_elbow_angle"] = np.clip(state["min_elbow_angle"], 30, 180)
        state["max_elbow_angle"] = np.clip(state["max_elbow_angle"], 30, 180)

        state["max_depth"] = np.clip(state["max_depth"], -2, 2)
        state["min_depth"] = np.clip(state["min_depth"], -2, 2)

        state["avg_alignment"] = np.clip(state["avg_alignment"], 120, 200)
        state["min_alignment"] = np.clip(state["min_alignment"], 120, 200)

        state["min_hip_angle"] = np.clip(state["min_hip_angle"], 120, 200)
        state["avg_hip_angle"] = np.clip(state["avg_hip_angle"], 120, 200)

        state["elbow_range"] = np.clip(state["elbow_range"], 0, 150)
        state["depth_range"] = np.clip(state["depth_range"], 0, 2)

        return state
    

    def step(self, state_vec, action):
        state = vector_to_dict(state_vec)
        new_state = state.copy()

        if np.random.rand() < self.success_prob:

            if action == 0:  # go deeper
                delta = np.random.uniform(0.05, 0.15)
                new_state["max_depth"] += delta
                new_state["depth_range"] += delta

                # tradeoff: worse alignment slightly
                new_state["avg_alignment"] -= np.random.uniform(0, 0.02)

            elif action == 1:  # fix elbow
                delta = np.random.uniform(3, 8)
                new_state["min_elbow_angle"] -= delta
                new_state["elbow_range"] += np.random.uniform(1, 3)

            elif action == 2:  # fix alignment
                delta = np.random.uniform(0.05, 0.15)
                new_state["avg_alignment"] += delta
                new_state["min_alignment"] += delta * 0.5

            elif action == 3:  # fix hip
                delta = np.random.uniform(2, 5)
                new_state["min_hip_angle"] += delta
                new_state["avg_hip_angle"] += delta * 0.5

        else:
            # failure / fatigue / randomness
            for k in new_state:
                new_state[k] += np.random.normal(0, 0.01)

        # natural noise always
        for k in new_state:
            new_state[k] += np.random.normal(0, 0.002)
        
        new_state = self.clip_state(new_state)

        return dict_to_vector(new_state)