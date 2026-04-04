from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from src.rl.env import PushupEnv

# Wrap environment with Monitor
env = Monitor(PushupEnv(), filename="logs")

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=50000)

model.save("pushup_rl_model")

print("✅ Training complete!")