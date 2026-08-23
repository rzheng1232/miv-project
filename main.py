import gymnasium as gym
import torch

def run_episode(env):
    """runs a single episode, returns reward"""
    obs, info = env.reset()
    done = False
    reward_tot = 0
    while not done:
        action = get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        reward_tot += reward
        done = terminated or truncated
    return reward_tot
def get_action(state):


def main():
    env = gym.make('Ant-v5')
    

main()