from simulation.pushup_simulator import PushupSimulator
import random

import matplotlib
matplotlib.use('Agg')  # or 'Qt5Agg'

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

plt.plot(errors)
plt.xlabel("Frame")
plt.ylabel("Error")
plt.title("Posture Error Over Time")

plt.savefig("error_plot.png")
plt.close()


# Plot error trend
#plt.plot(errors)
#plt.title("Error over Time")
#plt.xlabel("Steps")
#plt.ylabel("Posture Error")
#plt.show()