import numpy as np
from config import FEATURE_NAMES

def dict_to_vector(state_dict):
    return np.array([state_dict[f] for f in FEATURE_NAMES], dtype=np.float32)

def vector_to_dict(state_vector):
    return {f: state_vector[i] for i, f in enumerate(FEATURE_NAMES)}