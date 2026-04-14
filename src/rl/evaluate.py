import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from env import PushupEnv
from gym_env import GymPushupEnv


# -----------------------------
# LOAD DATA + MODELS
# -----------------------------
df = pd.read_csv("src/features.csv")

model_data = joblib.load("pushup_model.pkl")
clf_model = model_data["model"]

ppo_model = PPO.load("ppo_pushup")


# -----------------------------
# INIT ENV
# -----------------------------
env = PushupEnv(clf_model, df)
gym_env = GymPushupEnv(env)


# -----------------------------
# RUN EPISODE FUNCTION
# -----------------------------
def run_episode(policy="ppo", max_steps=25):
    state, _ = gym_env.reset()

    rewards = []
    scores = []
    actions = []

    for step in range(max_steps):

        # choose action
        if policy == "ppo":
            action, _ = ppo_model.predict(state)
        else:
            action = np.random.randint(0, 4)

        # step
        next_state, reward, done, _, _ = gym_env.step(action)

        # get score from underlying env
        score = env.get_score(next_state)

        rewards.append(reward)
        scores.append(score)
        actions.append(action)

        state = next_state

        if done:
            break

    return rewards, scores, actions


# -----------------------------
# RUN MULTIPLE EPISODES
# -----------------------------
def evaluate(n_episodes=10):
    ppo_scores = []
    rand_scores = []

    for _ in range(n_episodes):
        _, scores_ppo, _ = run_episode("ppo", 25)
        _, scores_rand, _ = run_episode("random", 25)

        ppo_scores.append(scores_ppo)
        rand_scores.append(scores_rand)

    return ppo_scores, rand_scores


# -----------------------------
# PLOT RESULTS
# -----------------------------
def plot_results(ppo_scores, rand_scores):

    plt.figure()

    # plot PPO
    for scores in ppo_scores:
        plt.plot(scores, linestyle='-', alpha=0.7)

    # plot Random
    for scores in rand_scores:
        plt.plot(scores, linestyle='--', alpha=0.5)

    plt.title("PPO vs Random: Score Improvement")
    plt.xlabel("Steps")
    plt.ylabel("Form Score")

    plt.show()


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    print("\n--- Running PPO Evaluation ---\n")

    ppo_scores, rand_scores = evaluate(n_episodes=7)

    plot_results(ppo_scores, rand_scores)

    print("\nSample PPO Episode:\n")
    rewards, scores, actions = run_episode("ppo")

    for i in range(len(actions)):
        print(f"Step {i} | Action: {actions[i]} | Reward: {rewards[i]:.3f} | Score: {scores[i]:.3f}")