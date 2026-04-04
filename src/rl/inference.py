from stable_baselines3 import PPO
from rl.env import PushupEnv

env = PushupEnv()
model = PPO.load("pushup_rl_model")

state = env.reset()

for _ in range(50):
    action, _ = model.predict(state)

    state, reward, done, _ = env.step(action)

    print("Action:", action, "Reward:", reward)

    if done:
        print("✅ Perfect posture achieved!")
        break