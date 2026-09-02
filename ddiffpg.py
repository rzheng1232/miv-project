import numpy as np
import torch.nn as nn
import torch
import random
import numpy as np
import params
from dataclasses import dataclass, field
from dtaidistance import dtw_ndim
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

def SinusoidalEmbedding(t, emb_len):
    #TODO
    """calculates the sinusoidal time embedding for timestep t"""
def BuildTransitionState(s_curr, a, s_next):
    """Calculates a_target"""

@dataclass
class TransitionState:
    
    s: torch.Tensor
    a: torch.Tensor
    a_target: torch.Tensor
    s_next: torch.Tensor
    r: torch.Tensor
    mode_id: int = -1

@dataclass
class TrajMode:
    id: int
    embedding: torch.Tensor
    prev_cluster: list[int] = field(default_factory=list)
    

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

class CriticNet(nn.Module):
    def __init__(self, StateDim, ActionDim, num_atoms=51):
        super().__init__()
        self.mlp = nn.Sequential(
                nn.Linear(StateDim + ActionDim, 512),
                nn.ELU(),
                nn.Linear(512, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
                nn.Linear(128, num_atoms),
            )

    def forward(self, state, action):
        return self.mlp(torch.cat([state, action], dim=-1))

class Critic():
    
    def __init__(self, StateDim, ActionDim):
        """Distributional Double Q Critic"""
        self.q1 = CriticNet(StateDim, ActionDim)
        self.q2 = CriticNet(StateDim, ActionDim)

    def getq1q2(self, state, action):
        return torch.softmax(self.q1.forward(state, action)), torch.softmax(self.q2.forward(state, action), dim=-1)
    def getqmin(self, state, action):
        Q1, Q2 = self.getq1q2(state, action)
        Q1 = torch.sum(Q1 * self.z_atoms.to(self.device), dim=1)
        Q2 = torch.sum(Q2 * self.z_atoms.to(self.device), dim=1)
        return torch.min(Q1, Q2)

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
        
    def get_actions(self, state, num_envs):
        # denoise action from random vector of dim (1, action_size)
        a = nn.randn(num_envs,self.out_size)
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
    def __init__(self, num_envs, lr):
        self.trajectories = [[] for _ in range(num_envs)] 
        self.finished = []
        self.dp = DiffusionPolicy() #TODO
        self.tot_updates = 0 # track progress of training
        self.num_envs = num_envs
        self.dist_matrix_cache = np.zeros((len(self.finished), len(self.finished)))
        self.traj_id = 0
        self.modes: dict[int, TrajMode] = {}
        self.Q_functions: dict[int, Critic] = {} 
        self.lr = lr
    def explore_env(self, env, num_timesteps, rand, total_steps):
        """Collect environment transitions into a buffer.
        Args:
            env: The MuJoCo environment.
            num_timesteps (int): Total timesteps to collect per buffer.
            rand (bool): If True, enables random actions for warmup phase.
            total_steps (int): Total number of updates completed in the training session.
                Used to schedule exploration decay.
        """
        observation, info = env.reset()
        t = 0
        while t < self.num_timesteps:
            if rand:
                action = env.action_space.sample() 
            else:
                # TODO: add mode embedding to training
                action = self.dp.get_actions(observation, self.num_envs)


            obs_prev = observation
            observation, reward, terminated, truncated, info = env.step(action)
            episode_over = np.logical_or(terminated, truncated)
            # obs, action, action_target=action (temp value), obs_next, reward 
            
            for i in range(self.num_envs):
                ts = TransitionState(obs_prev[i], action[i], action[i], observation[i], reward[i])
                self.trajectories[i].append(ts)
                if (episode_over[i]):
                    self.finished.append((self.trajectories[i], self.traj_id, -1))
                    self.traj_id+=1
                    self.trajectories[i] = []
            t+=1
        
    def sample_action(self, obs, mode_embedding):
        """sample the action from the diffusion policy
                Args:
                    obs - the current timestep observation of the agent to produce the action from
                    mode_embedding - determines what mode the diffusion policy should produce. explore_embedding or specific mode embedding 

                """
        state = obs + mode_embedding
        self.dp.get_actions(state, 1)
    def new_mode(self, mode_id, prev_cluster, parent_mode=None, noise_scale=0.2):
        """create a new mode embedding and add to the modes dict"""
        if (parent_mode is None):
            vec = torch.randn(params.embed_dim) * noise_scale
        else:
            vec = parent_mode.embedding.detach().clone() + torch.randn(params.embed_dim) * noise_scale
        traj_mode= TrajMode(id=mode_id, embedding=nn.Parameter(vec), prev_cluster=prev_cluster)
        self.modes[mode_id] = traj_mode
        return traj_mode
    
    def process_trajs(self):
        """Calculate non-cached dtw distances, cluster trajectories using distance matrix, calculate target actions"""
        N_old = len(self.dist_matrix_cache) if self.dist_matrix_cache is not None else 0
        N_new = len(self.finished)
        dist_matrix = np.zeros((N_new,N_new))
        if (N_old > 0):
            dist_matrix[:N_old, :N_old] = self.dist_matrix_cache    
        for i in range(N_old, N_new):
            for j in range(N_new):
                if (i == j): continue
                dist_matrix[i][j] = dtw_ndim.distance(self.finished[i], self.finished[j])
                dist_matrix[j][i] = dtw_ndim.distance(self.finished[i], self.finished[j])
        self.dist_matrix_cache = dist_matrix
        #clustering
        dist_matrix = squareform(dist_matrix)
        Z = linkage(dist_matrix, method='average')  
        threshold = 0.7 * max(Z[:,2]) 
        labels = fcluster(Z, t=threshold, criterion='distance') 
        num_clusters = len(set(labels))
        clusters = [[] for l in range(num_clusters)]
        for i in range(len(labels)):
            clusters[labels[i]-1].append(self.finished[i][1])
        mode_max_match_dict = {} # maps the mode to the cluster index with most match 
        cluster_best_group = []
        if (self.modes):
            # match this clusters back to ground truth cluster index
            for i in range(len(clusters)):
                max_matches = len(set(clusters[i]) & set(self.modes[0].prev_cluster))
                max_group = 0
                for j in range(1, len(self.modes)):
                    matches = len(set(clusters[i]) & set(self.modes[j].prev_cluster))
                    if (matches>max_matches):
                        max_matches = matches
                        max_group = j
                if max_group not in mode_max_match_dict or max_matches > mode_max_match_dict[max_group][1]:
                    mode_max_match_dict[max_group] = (i, max_matches)
                cluster_best_group.append((max_group, max_matches))
            for i in range(len(clusters)):
                if (cluster_best_group[i][1] == 0):
                    # new group, create new Q function,new embedding for mode
                    self.Q_functions[len(self.Q_functions)] = (Critic(stateDim=params.state_dim, ActionDim=params.action_dim))
                    self.new_mode(mode_id=len(self.Q_functions)-1, prev_cluster=clusters[i])
                else:
                    if mode_max_match_dict[cluster_best_group[i][0]][0] == i:
                        # largest matches, so this cluster inherits original Q function and mode embedding
                        self.modes[cluster_best_group[i][0]].prev_cluster = clusters[i]
                    else: 
                        parent_id = cluster_best_group[i][0]
                        parent_critic = self.Q_functions[parent_id]
                        cloned = Critic(stateDim=params.state_dim, ActionDim=params.action_dim)
                        cloned.q1.load_state_dict(parent_critic.q1.state_dict())
                        cloned.q2.load_state_dict(parent_critic.q2.state_dict())
                        self.Q_functions[len(self.Q_functions)] = cloned
                        self.new_mode(mode_id=len(self.Q_functions)-1, prev_cluster=clusters[i], parent_mode=self.modes[cluster_best_group[i][0]])
                 
        else:
            # if no clusters previously identified, then create new Q function for each cluster
            for i in range(len(clusters)):
                self.Q_functions[len(self.Q_functions)] = (Critic(stateDim=params.state_dim, ActionDim=params.action_dim))
                self.new_mode(mode_id=len(self.Q_functions)-1, prev_cluster=clusters[i])
        traj_to_prev_mode = {}
        for mode_id, mode in self.modes.items():
            for tid in mode.prev_cluster:
                traj_to_prev_mode[tid] = mode_id
        # calculate target action from action gradient
        for i in range(len(self.finished)):
            critic = self.Q_functions[traj_to_prev_mode[self.finished[i][1]]]
            for j in range(len(self.finished[i][0])):
                a = self.finished[i][0][j].a.clone().detach().requires_grad_(True)
                s = self.finished[i][0][j].s
                qmin = critic.getqmin(s, a)
                a_grad = torch.autograd.grad(outputs=qmin, inputs=a)[0]
                a_target = self.finished[i][0][j].a + self.lr * a_grad
                self.finished[i][0][j].a_target = a_target
        
        return dist_matrix

    def update_policy(self):
        """Update diffusion policy weights indirectly using target action and behavorial cloning objective """
        loss = self.dp.get_loss(self.finished)
        loss.backward()
    

    