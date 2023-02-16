from IPython.display import clear_output
import numpy as np
import random
import gymnasium as gym
import time
import matplotlib.pyplot as plt

environment = gym.make("FrozenLake-v1", is_slippery=True)
environment.reset()
environment.render()

# Our table has the following dimensions:
# (rows x columns) = (states x actions) = (16 x 4)
qtable = np.zeros((16, 4))

# Alternatively, the gym library can also directly g
# give us the number of states and actions using
# "env.observation_space.n" and "env.action_space.n"
nb_states = environment.observation_space.n  # = 16
nb_actions = environment.action_space.n  # = 4
qtable = np.zeros((nb_states, nb_actions))

# Let's see how it looks
print("Q-table =")
print(qtable)

action = environment.action_space.sample()

# 2. Implement this action and move the agent in the desired direction
new_state, reward, terminated, info, truncated = environment.step(action)

# Display the results (reward and map)
environment.render()
print(f"Reward = {reward}")

plt.rcParams["figure.dpi"] = 300
plt.rcParams.update({"font.size": 17})

# We re-initialize the Q-table
qtable = np.zeros((environment.observation_space.n, environment.action_space.n))

# Hyperparameters
episodes = 1000  # Total number of episodes
alpha = 0.5  # Learning rate
gamma = 0.9  # Discount factor

# List of outcomes to plot
outcomes = []

print("Q-table before training:")
print(qtable)

# Training
for _ in range(episodes):
    state = environment.reset()[0]
    terminated = False

    # By default, we consider our outcome to be a failure
    outcomes.append("Failure")

    # Until the agent gets stuck in a hole or reaches the goal, keep training it
    while not terminated:
        # Choose the action with the highest value in the current state
        print(int(np.max(qtable[0])))
        if np.max(int(qtable[state][0])) > 0:
          print("popo")
          action = np.argmax(int(qtable[state[0]]))

        # If there's no best action (only zeros), take a random one
        else:
          action = environment.action_space.sample()
             
        # Implement this action and move the agent in the desired direction
        onservation, reward, terminated, truncated, info = environment.step(action)

        # Update Q(s,a)
        print(state)
        qtable[int(state), action] = qtable[state, action] + \
                                alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])
        
        # Update our current state
        state = onservation
        
        # If we have a reward, it means that our outcome is a success
        if reward:
          outcomes[-1] = "Success"

print()
print('===========================================')
print('Q-table after training:')
print(qtable)
print("aled")
# Plot outcomes
plt.figure(figsize=(12, 5))
plt.xlabel("Run number")
plt.ylabel("Outcome")
ax = plt.gca()
ax.set_facecolor('#efeeea')
plt.bar(range(len(outcomes)), outcomes, color="#0A047A", width=1.0)
plt.show()
