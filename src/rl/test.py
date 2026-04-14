import numpy as np
import pandas as pd
import joblib

from env import PushupEnv

# Load dataset
df = pd.read_csv("src/features.csv")

# Load trained model
model_data = joblib.load("pushup_model.pkl")
model = model_data["model"]

# Initialize env
env = PushupEnv(model, df)

state = env.reset()
print("Initial State:", state)

print("\n--- Running Random Actions ---\n")

for step in range(10):
    action = np.random.randint(0, 4)

    next_state, reward, done, _ = env.step(action)

    print(f"Step {step}")
    print("Action:", action)
    print("Reward:", reward)

    if done:
        print("\nEpisode finished early\n")
        break