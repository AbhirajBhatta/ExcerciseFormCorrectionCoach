from simulation.pushup_simulator import PushupSimulator
import random
import matplotlib.pyplot as plt

env = PushupSimulator()

state = env.reset()
print("Initial State:", state)

errors = []

for step in range(50):
    action = random.randint(0, 2)

    next_state, reward, done = env.step(action)

    error = env.compute_error()
    errors.append(error)

    print(f"\nStep {step}")
    print("Action:", action)
    print("State:", next_state)
    print("Reward:", reward)
    print("Error:", error)

    if done:
        print("\n✅ Reached good posture!")
        break

# Plot error trend
plt.plot(errors)
plt.title("Error over Time")
plt.xlabel("Steps")
plt.ylabel("Posture Error")
plt.show()