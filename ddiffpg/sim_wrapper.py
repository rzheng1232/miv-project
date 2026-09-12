import gymnasium as gym
import numpy as np
class AntGaitWrapper(gym.Wrapper):
    def __init__(self, env, success_dist=8.0, target_vel=2.5, success_radius=0.5):
        super().__init__(env)
        self.success_dist = success_dist
        self.success_radius = success_radius
        self.target_vel = target_vel

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.x0 = self.env.unwrapped.data.qpos[0]
        self.ever_succeeded = False
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        x = self.env.unwrapped.data.qpos[0]
        dist_remaining = self.success_dist - (x - self.x0)
        vel_x = self.env.unwrapped.data.qvel[0]

        reward = -abs(vel_x - self.target_vel)
        self.ever_succeeded = self.ever_succeeded or (dist_remaining <= self.success_radius)
        info["success"] = self.ever_succeeded
        info["dist_remaining"] = dist_remaining

        return obs, reward, terminated, truncated, info   