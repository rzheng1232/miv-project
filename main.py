import gymnasium as gym
from ddiffpg import ddiffpg
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
    env = gym.make_vec("Ant-v5", num_envs=256, vectorization_mode="sync")
    m = ddiffpg(num_envs=256, lr=0.1)
    m.explore_env(env, num_timesteps=1000)
    m.process_trajs()
    m.update_policy()

main()