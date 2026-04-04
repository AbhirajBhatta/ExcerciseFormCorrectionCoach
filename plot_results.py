import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("logs/monitor.csv", skiprows=1)

# Extract reward and timesteps
rewards = df["r"]
timesteps = df["l"].cumsum()

# Plot
plt.plot(timesteps, rewards)
plt.xlabel("Timesteps")
plt.ylabel("Reward")
plt.title("RL Learning Curve")
plt.show()