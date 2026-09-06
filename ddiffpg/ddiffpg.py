import numpy as np
import torch.nn as nn
import torch.nn.functional as f
import torch
import random
import numpy as np
import ddiffpg.params as params
from dataclasses import dataclass, field
from dtaidistance import dtw_ndim
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

def SinusoidalEmbedding(t, emb_len):
    """calculates the sinusoidal time embedding for timestep t (t is a scalar diffusion step)"""
    half_dim = emb_len // 2
    freqs = torch.exp(-np.log(10000.0) * torch.arange(half_dim, dtype=torch.float32) / half_dim)
    t = torch.as_tensor(t, dtype=torch.float32).reshape(())
    args = t * freqs
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb_len % 2 == 1:
        emb = torch.cat([emb, torch.zeros(1)], dim=-1)
    return emb

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
        t_emb = self.t_mlp(SinusoidalEmbedding(t, self.t_emb_size))
        if x.dim() > 1:
            # t is a single scalar diffusion step shared by the whole batch; broadcast it to match
            t_emb = t_emb.unsqueeze(0).expand(x.shape[0], -1)
        return self.mlp(torch.cat([t_emb, state, x], dim=-1))

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

    def __init__(self, StateDim, ActionDim, num_atoms=51, v_min=0, v_max=5, tau=0.05, gamma=0.99):
        """Distributional Double Q Critic, with buffered (target) copies of Q1, Q2, and the actor
        used only to construct Bellman targets - never trained directly by backprop."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tau = tau
        self.z_atoms = torch.linspace(v_min, v_max, num_atoms).to(self.device)
        self.gamma = torch.tensor(gamma, device=self.device)
        self.q1 = CriticNet(StateDim, ActionDim, num_atoms).to(self.device)
        self.q2 = CriticNet(StateDim, ActionDim, num_atoms).to(self.device)
        self.target_q1 = CriticNet(StateDim, ActionDim, num_atoms).to(self.device)
        self.target_q2 = CriticNet(StateDim, ActionDim, num_atoms).to(self.device)
        self._hard_copy(self.target_q1, self.q1)
        self._hard_copy(self.target_q2, self.q2)

        self.buffered_actor = DiffusionPolicy(state_size=params.state_dim, action_size=params.action_dim,
                                           num_steps=params.diffusion_steps, beta=params.beta)
        self._actor_initialized = False  # buffered_Actor has no live actor to copy from until update_buffered() runs

    @staticmethod
    def _hard_copy(target_net, online_net):
        target_net.load_state_dict(online_net.state_dict())
        for p in target_net.parameters():
            p.requires_grad = False

    def _soft_update(self, target_net, online_net):
        for t_param, o_param in zip(target_net.parameters(), online_net.parameters()):
            t_param.data.mul_(1 - self.tau).add_(o_param.data, alpha=self.tau)

    def update_buffered(self, updated_actor):
        """Polyak-update the buffered Q-functions, and the buffered actor, toward their live counterparts."""
        self._soft_update(self.target_q1, self.q1)
        self._soft_update(self.target_q2, self.q2)
        if not self._actor_initialized:
            self._hard_copy(self.buffered_actor, updated_actor)
            self._actor_initialized = True
        else:
            self._soft_update(self.buffered_actor, updated_actor)

    def getq1q2(self, state, action, online=True):
        if (online):
            return torch.softmax(self.q1.forward(state, action), dim=-1), torch.softmax(self.q2.forward(state, action), dim=-1)
        else:
            return torch.softmax(self.target_q1.forward(state, action), dim=-1), torch.softmax(self.target_q2.forward(state, action), dim=-1)
    def getqmin(self, state, action, online=True, scalar=True):
        Q1, Q2 = self.getq1q2(state, action, online=online)
        E1 = torch.sum(Q1 * self.z_atoms, dim=-1)
        E2 = torch.sum(Q2 * self.z_atoms, dim=-1)
        if (scalar):
            return torch.min(E1, E2)
        else:
            use_q1 = (E1 <= E2).unsqueeze(-1)  
            return torch.where(use_q1, Q1, Q2)
    def get_loss(self, states_samples, actions_samples, next_states, rewards):
        """bellman update loss for distributional double Q learning"""
        pred = self.getqmin(states_samples, actions_samples, online=True, scalar=False)
        next_action = self.buffered_actor.get_actions(next_states, next_states.shape[0])
        bellman_expected_reward = rewards + self.gamma * self.getqmin(next_states, next_action, online=False, scalar=False)
        loss = f.cross_entropy(pred, bellman_expected_reward)
        return loss
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
        a = torch.randn(num_envs, self.out_size)
        for k in range(1, self.T):
            t =  self.T - k
            noise_pred = self.model.forward(a, state, t)
            a = (a - noise_pred * torch.sqrt(self.betas[t])) / torch.sqrt(1-self.betas[t])
        return a;
        
    def get_loss(self, traj_samples):
        # assume one transition state sample is a struct that contains (s, a, a_target, s', r)
        loss_tot = 0
        for sample in traj_samples:
            t = random.randint(1, self.T - 1)
            eps = torch.randn_like(sample.a)
            # key: train to diffuse torwards a_target instead of a
            noised_act = torch.sqrt(self.alpha_bars[t])*sample.a_target + eps * torch.sqrt(1-self.alpha_bars[t])
            eps_pred = self.model.forward(noised_act, sample.s, t)
            this_loss = torch.dist(eps, eps_pred)
            loss_tot += this_loss
        loss_avg = loss_tot / len(traj_samples)
        return loss_avg
        

class ddiffpg():
    def __init__(self, num_envs, lr):
        self.trajectories = [[] for _ in range(num_envs)]
        self.success_trajs = []
        self.fail_trajs = []
        self.dp = DiffusionPolicy(state_size=params.state_dim, action_size=params.action_dim,
                                   num_steps=params.diffusion_steps, beta=params.beta)
        self.tot_updates = 0 # track progress of training
        self.num_envs = num_envs
        self.dist_matrix_cache = np.zeros((0, 0))
        self.traj_id = 0
        self.modes: dict[int, TrajMode] = {}
        self.Q_functions: dict[int, Critic] = {}
        self.lr = lr
        self.target_vel = [1.0] * num_envs  # TODO: set/sample real per-episode target velocities
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
        success_streak = [0 for _ in range(self.num_envs)]
        episode_len = [0 for _ in range(self.num_envs)]
        while t < num_timesteps:

            if rand:
                action = env.action_space.sample()
            else:
                # TODO: add mode embedding to training
                action = self.dp.get_actions(observation, self.num_envs)


            obs_prev = observation
            observation, reward, terminated, truncated, info = env.step(action)
            episode_over = np.logical_or(terminated, truncated)

            for i in range(self.num_envs):
                episode_len[i] += 1
                v_x = info["x_velocity"][i]
                if abs(v_x - self.target_vel[i]) <= 0.05:
                    success_streak[i] += 1

                ts = TransitionState(obs_prev[i], action[i], action[i], observation[i], reward[i])
                self.trajectories[i].append(ts)
                if (episode_over[i]):
                    if (terminated[i]):
                        self.fail_trajs.append((self.trajectories[i], self.traj_id))
                    elif (success_streak[i]/episode_len[i] > 0.5):
                        self.success_trajs.append((self.trajectories[i], self.traj_id))
                    else:
                        self.fail_trajs.append((self.trajectories[i], self.traj_id))
                    self.traj_id+=1
                    self.trajectories[i] = []
                    success_streak[i] = 0
                    episode_len[i] = 0
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

    def assign_unsuccessful_trajs(self, clusters, N=5):
        """Assign each unsuccessful trajectory to the
        cluster with the smallest average DTW distance to N trajectories sampled
        from that cluster."""
        if not clusters:
            return clusters
        traj_by_id = {tid: traj for traj, tid in self.success_trajs}
        for traj, traj_id in self.fail_trajs:
            query = [ts.s for ts in traj]
            best_idx, best_avg_dist = None, float('inf')
            for idx, cluster in enumerate(clusters):
                if not cluster:
                    continue
                sample_ids = random.sample(cluster, min(N, len(cluster)))
                dists = [dtw_ndim.distance(query, [ts.s for ts in traj_by_id[sid]]) for sid in sample_ids]
                avg_dist = sum(dists) / len(dists)
                if avg_dist < best_avg_dist:
                    best_avg_dist = avg_dist
                    best_idx = idx
            if best_idx is not None:
                clusters[best_idx].append(traj_id)
        return clusters

    def process_trajs(self):
        """Calculate non-cached dtw distances, cluster trajectories using distance matrix, calculate target actions"""
        N_old = len(self.dist_matrix_cache) if self.dist_matrix_cache is not None else 0
        N_new = len(self.success_trajs)
        dist_matrix = np.zeros((N_new,N_new))
        if (N_old > 0):
            dist_matrix[:N_old, :N_old] = self.dist_matrix_cache
        for i in range(N_old, N_new):
            for j in range(N_new):
                if (i == j): continue
                states_i = [ts.s for ts in self.success_trajs[i][0]]
                states_j = [ts.s for ts in self.success_trajs[j][0]]
                d = dtw_ndim.distance(states_i, states_j)
                dist_matrix[i][j] = d
                dist_matrix[j][i] = d
        self.dist_matrix_cache = dist_matrix
        #clustering
        dist_matrix = squareform(dist_matrix)
        Z = linkage(dist_matrix, method='average')  
        threshold = 0.7 * max(Z[:,2]) 
        labels = fcluster(Z, t=threshold, criterion='distance') 
        num_clusters = len(set(labels))
        clusters = [[] for l in range(num_clusters)]
        for i in range(len(labels)):
            clusters[labels[i]-1].append(self.success_trajs[i][1])
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
                    self.Q_functions[len(self.Q_functions)] = (Critic(StateDim=params.state_dim, ActionDim=params.action_dim))
                    self.new_mode(mode_id=len(self.Q_functions)-1, prev_cluster=clusters[i])
                else:
                    if mode_max_match_dict[cluster_best_group[i][0]][0] == i:
                        # largest matches, so this cluster inherits original Q function and mode embedding
                        self.modes[cluster_best_group[i][0]].prev_cluster = clusters[i]
                    else: 
                        parent_id = cluster_best_group[i][0]
                        parent_critic = self.Q_functions[parent_id]
                        cloned = Critic(StateDim=params.state_dim, ActionDim=params.action_dim)
                        cloned.q1.load_state_dict(parent_critic.q1.state_dict())
                        cloned.q2.load_state_dict(parent_critic.q2.state_dict())
                        self.Q_functions[len(self.Q_functions)] = cloned
                        self.new_mode(mode_id=len(self.Q_functions)-1, prev_cluster=clusters[i], parent_mode=self.modes[cluster_best_group[i][0]])
                 
        else:
            # if no clusters previously identified, then create new Q function for each cluster
            for i in range(len(clusters)):
                self.Q_functions[len(self.Q_functions)] = (Critic(StateDim=params.state_dim, ActionDim=params.action_dim))
                self.new_mode(mode_id=len(self.Q_functions)-1, prev_cluster=clusters[i])

        clusters = self.assign_unsuccessful_trajs(clusters)

        traj_to_prev_mode = {}
        for mode_id, mode in self.modes.items():
            for tid in mode.prev_cluster:
                traj_to_prev_mode[tid] = mode_id
        # calculate target action from action gradient, for both successful and unsuccessful trajectories
        for traj_list in (self.success_trajs, self.fail_trajs):
            for traj, traj_id in traj_list:
                mode_id = traj_to_prev_mode.get(traj_id)
                if mode_id is None:
                    continue
                critic = self.Q_functions[mode_id]
                for ts in traj:
                    a = ts.a.clone().detach().requires_grad_(True)
                    qmin = critic.getqmin(ts.s, a)
                    a_grad = torch.autograd.grad(outputs=qmin, inputs=a)[0]
                    ts.a_target = ts.a + self.lr * a_grad

        return dist_matrix
    def _mode_transitions(self, mode_id):
        """Flatten every transition belonging to trajectories currently assigned to mode_id."""
        traj_by_id = {tid: traj for traj, tid in self.success_trajs + self.fail_trajs}
        transitions = []
        for tid in self.modes[mode_id].prev_cluster:
            transitions.extend(traj_by_id[tid])
        return transitions

    def build_batch(self, batch_size=4096):
        size_per_mode = batch_size // len(self.modes)
        batch = []
        for m_id in self.modes:
            pool = self._mode_transitions(m_id)
            if not pool:
                continue
            batch += random.choices(pool, k=size_per_mode)
        return batch
    def update_critic(self):
        for mode_id in self.modes:
            pool = self._mode_transitions(mode_id)
            if not pool:
                continue
            training_batch = random.choices(pool, k=2000)
            state = torch.stack([sample.s for sample in training_batch])
            action = torch.stack([sample.a for sample in training_batch])
            next_states = torch.stack([sample.s_next for sample in training_batch])
            rewards = torch.stack([sample.r for sample in training_batch])
            critic = self.Q_functions[mode_id]
            loss = critic.get_loss(state, action, next_states, rewards)

            # TODO: Bellman target (reward + buffered actor/critic bootstrap), critic.get_loss,
            # backward + optimizer step, then critic.update_buffered(self.dp.model)

    def update_policy(self):
        """Update diffusion policy weights indirectly using target action and behavorial cloning objective """
        training_batch = self.build_batch()
        loss = self.dp.get_loss(training_batch)
        loss.backward()
    

    