import pandas as pd
import joblib

from stable_baselines3 import PPO

from env import PushupEnv
from gym_env import GymPushupEnv

# Load dataset
df = pd.read_csv("src/features.csv")

X = df.drop(columns=["label", "video_path"])
y = df["label"]

# Load model (you commented saving — FIX THAT FIRST)
model_data = joblib.load("pushup_model.pkl")
model = model_data["model"]

env = PushupEnv(model, df)
gym_env = GymPushupEnv(env)

model_rl = PPO(
    "MlpPolicy",
    gym_env,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log="./ppo_logs/"
)

model_rl.learn(total_timesteps=50000)

model_rl.save("ppo_pushup")