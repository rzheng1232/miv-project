import torch
import gymnasium as gym
# test run for a gymnasium episode

env = gym.make('Ant-v5', render_mode="human")
episode_over = False
total_reward = 0
observation, info = env.reset()
while not episode_over:
    action = env.action_space.sample() 
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    episode_over = terminated or truncated
print(f"Episode finished! Total reward: {total_reward}")
env.close()