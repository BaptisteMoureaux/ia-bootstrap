import gymnasium as gym
env = gym.make('FrozenLake-v1', render_mode='human', is_slippery=True)
env.reset()
t = 0
while True:
    action = env.action_space.sample()
    new_state, reward, terminated, info, truncated = env.step(action)
    print(f'Reward = {reward}')
    env.render()  
    t += 1
    if terminated:
        print("Episode finished after {} timesteps".format(t+1))
        env.reset()
        t = 0