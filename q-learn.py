import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

environment = gym.make("FrozenLake-v1")
# environment = gym.make("FrozenLake-v1", render_mode="human", is_slippery=True)
environment.reset()

# We re-initialize the Q-table
qtable = np.zeros((environment.observation_space.n, environment.action_space.n))

# Hyperparameters
episodes = 30000        # Total number of episodes
alpha = 0.9        # Learning rate
gamma = 0.9            # Discount factor
epsilon = 1.0          # Amount of randomness in the action selection
epsilon_decay = 0.001  # Fixed amount to decrease

# List of outcomes to plot
outcomes = []

# Training
start_time = time.time() # Start time of training
for i in range(episodes):
    state = environment.reset()
    done = False

    # By default, we consider our outcome to be a failure
    outcomes.append("Failure")

    # Until the agent gets stuck in a hole or reaches the goal, keep training it
    while not done:
        # Generate a random number between 0 and 1
        rnd = np.random.random()

        # If random number < epsilon, take a random action
        if rnd < epsilon:
            action = environment.action_space.sample()
        # Else, take the action with the highest value in the current state
        else:
            action = np.argmax(qtable[state[0]])

        # Implement this action and move the agent in the desired direction
        observation, reward, terminated, truncated, info = environment.step(action)

        # Update Q(s,a)
        qtable[state[0], action] = qtable[state[0], action] + \
            alpha * (reward + gamma *
                     np.max(qtable[observation]) - qtable[state[0], action])

        # Update our current state
        state = (observation,)

        # If we have a reward, it means that our outcome is a success
        if reward:
            outcomes[-1] = "Success"
            
        done = terminated or truncated

    # Update epsilon
    epsilon = max(epsilon - epsilon_decay, 0)

    # Calculate and print progress
    if i % 1000 == 0:
        print(f"Training progress: {i/episodes*100:.2f}%", end="\r")

training_time = time.time() - start_time # Total training time
print(f"Training time: {training_time:.2f} seconds")

# Evaluation
start_time = time.time() # Start time of evaluation
episodes = 1000
nb_success = 0


environment = gym.make("FrozenLake-v1", render_mode="human")
# environment = gym.make("FrozenLake-v1", render_mode="human", is_slippery=True)

for i in range(episodes):
    state = environment.reset()
    terminated = False

    # Until the agent gets stuck or reaches the goal, keep training it
    while not terminated:
        # Choose the action with the highest value in the current state
        action = np.argmax(qtable[state[0]])

        # Implement this action and move the agent in the desired direction
        observation, reward, terminated, truncated, info = environment.step(action)

        # Update our current state
        state = (observation,)

        # When we get a reward, it means we solved the game
        nb_success += reward

    # Calculate and print progress
    if i % 100 == 0:
        print(f"Evaluation progress: {i/episodes*100:.2f}%", end="\r")

evaluation_time = time.time() - start_time # Total training time
print(f"Evaluation time: {evaluation_time:.2f} seconds")
print(f"Success rate = {nb_success/episodes*100}%")

#? Pourquoi le success rate n'est pas de 100% en non slippery mdoe ?
