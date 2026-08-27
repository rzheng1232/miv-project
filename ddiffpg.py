import torch.nn as nn
import torch
import random
import params
from dataclasses import dataclass, field

def SinusoidalEmbedding(t, emb_len):
    #TODO
    """calculates the sinusoidal time embedding for timestep t"""
@dataclass
class TransitionState:
    s: torch.Tensor
    a: torch.Tensor
    a_target: torch.Tensor
    s_next: torch.Tensor
    r: torch.Tensor

class DiffusionNet(nn.Module):
    def __init__(self, t_emb_size, activation_fn=nn.Mish()):
        """Initialization for DiffusionNET. 
                Specify the embedding size for the time, and optionally specify activation function (default nn.Mish())"""
        super().__init__()
        self.mlp = nn.Sequential(
                nn.Linear(params.embed_dim + params.transition_dim, 1024),
                activation_fn,
                nn.Linear(1024, 512),
                activation_fn,
                nn.Linear(512, 256),
                activation_fn,
                nn.Linear(256, params.action_dim),
            )
        # we can use a time mlp to learn best way to represent the time signal to condition the diffusion
        self.t_mlp = nn.Sequential(
                nn.Linear(t_emb_size, t_emb_size*4),
                activation_fn,
                nn.Linear(t_emb_size*4, t_emb_size)
            )
        self.t_emb_size = t_emb_size


    def forward(self, x, state, t):
        # forward process
        return self.mlp(
            torch.cat([self.t_mlp(SinusoidalEmbedding(t, self.t_emb_size)), state, x], dim=-1))


class DiffusionPolicy():
    def __init__(self, state_size, action_size, num_steps, beta):
        self.model = DiffusionNet(t_emb_size = 256)
        self.T = num_steps
        self.in_size = state_size
        self.out_size = action_size
        self.beta_strt = 1e-4 # from ddpm original paper
        self.beta_end = 2e-2 
        self.betas = torch.linspace(self.beta_strt, self.beta_end, self.T)
        self.alpha_bars = torch.cumprod(1 - self.betas, dim=0)
        
    def get_actions(self, state, sample):
        # denoise action from random vector of dim (1, action_size)
        a = nn.randn(1,self.out_size)
        for k in range(1, self.T):
            t =  self.T - k
            noise_pred = self.model.forward(a, state, t)
            a = (a - noise_pred * torch.sqrt(self.betas[t])) / torch.sqrt(1-self.betas[t])
        return a;
        
    def get_loss(self, traj_samples):
        # assume one transition state sample is a struct that contains (s, a, a_target, s', r)
        loss_tot = 0
        for sample in traj_samples:
            t = random.randint(1, self.T)
            eps = random.random()
            # key: train to diffuse torwards a_target instead of a
            noised_act = torch.sqrt(self.alpha_bars[t])*sample.a_target + eps * torch.sqrt(1-self.alpha_bars[t])
            eps_pred = self.model.forward(noised_act, sample.state, t)
            this_loss = torch.dist(eps, eps_pred)
            loss_tot += this_loss
        loss_avg = loss_tot / len(traj_samples)
        return loss_avg
        

class ddiffpg():
    def __init__(self):
        self.diffusion_buffer = []
        self.dp = DiffusionPolicy()
        self.tot_updates = 0 # track progress of training
        


    def explore_env(self, env, num_timesteps, rand, total_steps):
        """Collect environment transitions into a buffer.
        Args:
            env: The MuJoCo environment instance.
            num_timesteps (int): Total timesteps to collect per buffer.
            rand (bool): If True, enables random actions for warmup phase.
            total_steps (int): Total number of updates completed in the training session.
                Used to schedule exploration decay.
        """
        t = 0
        while t < num_timesteps:

            t+=1
    def sample_action(self, obs, mode_embedding):
        """sample the action from the diffusion policy
                Args:
                    obs - the current timestep observation of the agent to produce the action from
                    mode_embedding - determines what mode the diffusion policy should produce. explore_embedding or specific mode embedding 

                """
        # Generate random action vector (noise) at start
        
        # cocatenate the embedding for mode and the obs onto the observation vector

        # run denoising process 1...T

        # return action 
    
    def update_policy():
        """Update diffusion policy weights indirectly using target action and behavorial cloning objective """
        
    
    
    def update_critic():
        """"""
    def forward_critic():
        """d"""
    