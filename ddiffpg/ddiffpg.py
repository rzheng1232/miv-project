import numpy as np
import torch.nn as nn
import torch.nn.functional as f
import torch
import random
import numpy as np
import params 
from dataclasses import dataclass, field
from dtaidistance import dtw_ndim
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

class RunningMeanStd:
    """Welford/Chan online mean+variance estimate, used to normalize RND intrinsic
    rewards by their running std (scale only, no mean-centering - standard RND practice,
    since centering could push an otherwise-nonnegative novelty signal negative)."""
    def __init__(self, eps=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, x):
        x = torch.as_tensor(x, dtype=torch.float32).flatten()
        batch_count = x.numel()
        batch_mean = x.mean().item()
        batch_var = ((x - batch_mean) ** 2).mean().item()  # biased; well-defined even for batch_count=1

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        return x / (self.var ** 0.5 + 1e-8)

def SinusoidalEmbedding(t, emb_len, device=None):
    """calculates the sinusoidal time embedding for timestep t (t is a scalar diffusion step)"""
    half_dim = emb_len // 2
    freqs = torch.exp(-np.log(10000.0) * torch.arange(half_dim, dtype=torch.float32, device=device) / half_dim)
    t = torch.as_tensor(t, dtype=torch.float32, device=device).reshape(())
    args = t * freqs
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb_len % 2 == 1:
        emb = torch.cat([emb, torch.zeros(1, device=device)], dim=-1)
    return emb

@dataclass
class TransitionState:
    
    s: torch.Tensor
    a: torch.Tensor
    a_target: torch.Tensor
    s_next: torch.Tensor
    r: torch.Tensor
    r_intrinsic: torch.Tensor
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
        t_emb = self.t_mlp(SinusoidalEmbedding(t, self.t_emb_size, device=x.device))
        if x.dim() > 1:
            # t is a single scalar diffusion step shared by the whole batch; broadcast it to match
            t_emb = t_emb.unsqueeze(0).expand(x.shape[0], -1)
        return self.mlp(torch.cat([t_emb, state, x], dim=-1))

class CriticNet(nn.Module):
    def __init__(self, StateDim, ActionDim, num_atoms=51, lr=5e-4):
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
        self.optimizer =  torch.optim.AdamW(self.mlp.parameters(), lr=lr)


    def forward(self, state, action):
        return self.mlp(torch.cat([state, action], dim=-1))
    def update(self):
        self.optimizer.step()
class Critic():

    def __init__(self, StateDim, ActionDim, num_atoms=51, val_min=0, val_max=5, tau=0.05, gamma=0.99):
        """Distributional Double Q Critic, with buffered (target) copies of Q1, Q2, and the actor
        used only to construct Bellman targets - never trained directly by backprop."""
        self.device = get_device()
        self.tau = tau
        self.val_min=val_min
        self.val_max = val_max
        self.z_atoms = torch.linspace(val_min, val_max, num_atoms).to(self.device)
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
            self._hard_copy(self.buffered_actor.model, updated_actor.model)
            self._actor_initialized = True
        else:
            self._soft_update(self.buffered_actor.model, updated_actor.model)

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
    def get_loss(self, states_samples, actions_samples, next_states, rewards, mode_embedding):
        """bellman update loss for distributional double Q learning"""
        q1, q2 = self.getq1q2(states_samples, actions_samples, online=True)
        with torch.no_grad():
            mode_emb_batch = mode_embedding.expand(next_states.shape[0], -1)
            next_action = self.buffered_actor.get_actions(torch.cat([next_states, mode_emb_batch], dim=-1), next_states.shape[0])
            target_prb = self.getqmin(next_states, next_action, online=False, scalar=False)
            Tz = (rewards.unsqueeze(-1) + self.gamma * self.z_atoms).clamp(self.val_min, self.val_max)
            delta_z = (self.z_atoms[1] - self.z_atoms[0])
            b = (Tz - self.val_min) / delta_z
            lower = torch.floor(b).long()
            upper = (lower + 1).clamp(max=len(self.z_atoms) - 1)
            m = torch.zeros_like(target_prb)
            w_u = b - lower.float()                
            w_l = 1.0 - w_u
            m.scatter_add_(-1, lower, target_prb * w_l)
            m.scatter_add_(-1, upper, target_prb * w_u)

        loss  = -(m * torch.log(q1 + 1e-8)).sum(-1).mean() - (m * torch.log(q2 + 1e-8)).sum(-1).mean()

        return loss
    def update(self):
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), max_norm=1.0)
        self.q1.update()
        self.q2.update()
class DiffusionPolicy():
    def __init__(self, state_size, action_size, num_steps, beta, lr=3e-4):
        self.device = get_device()
        self.model = DiffusionNet(t_emb_size = 256).to(self.device)
        self.optimizer =  torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.T = num_steps
        self.in_size = state_size
        self.out_size = action_size
        self.beta_strt = 1e-4 # from ddpm original paper
        self.beta_end = 2e-2
        self.betas = torch.linspace(self.beta_strt, self.beta_end, self.T).to(self.device)
        self.alpha_bars = torch.cumprod(1 - self.betas, dim=0)

    def get_actions(self, state, num_envs):
        # denoise action from random vector of dim (1, action_size)
        a = torch.randn(num_envs, self.out_size, device=self.device)
        for k in range(1, self.T):
            t =  self.T - k
            noise_pred = self.model.forward(a, state, t)
            a = (a - noise_pred * self.betas[t] / torch.sqrt(1-self.alpha_bars[t])) / torch.sqrt(1-self.betas[t])

                                   
        return a;
        
    def get_loss(self, traj_samples):
        # traj_samples is a list of (sample, mode_embedding) pairs - see build_batch
        loss_tot = 0
        for sample, mode_embedding in traj_samples:
            t = random.randint(1, self.T - 1)
            eps = torch.randn_like(sample.a)
            # key: train to diffuse torwards a_target instead of a
            noised_act = torch.sqrt(self.alpha_bars[t])*sample.a_target + eps * torch.sqrt(1-self.alpha_bars[t])
            state = torch.cat([sample.s, mode_embedding])
            eps_pred = self.model.forward(noised_act, state, t)
            this_loss = torch.dist(eps, eps_pred)
            loss_tot += this_loss
        loss_avg = loss_tot / len(traj_samples)
        return loss_avg
    def update(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

class ddiffpg():
    def __init__(self, num_envs, lr):
        self.device = get_device()
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
        self.target_vel = [1.0] * num_envs  #: set/sample real per-episode target velocities
        self.mode_explore_embedding = nn.Parameter(torch.randn(params.embed_dim, device=self.device) * 0.2)
        self.rnd_reward_rms = RunningMeanStd()
        # q_explore's reward (normalized RND error) has a different scale/distribution than
        # bounded task reward, and with gamma close to 1 the discounted return can sum much
        # higher than a per-step ~unit-scale reward suggests - give it a wider atom range
        # than the task-reward critics' default [0, 5]. Revisit once real rnd_loss/intrinsic
        # reward magnitudes are visible from an actual run.
        self.q_explore = Critic(StateDim=params.state_dim, ActionDim=params.action_dim, val_min=0, val_max=20)
        self.pred_exp_r_mlp = nn.Sequential(nn.Linear(params.state_dim, 512), nn.ELU(),
                                        nn.Linear(512, 256), nn.ELU(),
                                        nn.Linear(256, 128), nn.ELU(),
                                        nn.Linear(128, 128)).to(self.device)
        self.pred_exp_r_target = nn.Sequential(nn.Linear(params.state_dim, 512), nn.ELU(),
                                     nn.Linear(512, 256), nn.ELU(),
                                     nn.Linear(256, 128), nn.ELU(),
                                     nn.Linear(128, 128)).to(self.device)
        self.exp_r_opt =  torch.optim.AdamW(self.pred_exp_r_mlp.parameters(), lr=lr)
        for layer in self.pred_exp_r_mlp:          # nn.Sequential is iterable — yields Linear, ELU, Linear, ELU, ...
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)
        for layer in self.pred_exp_r_target:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)
        for param in self.pred_exp_r_target.parameters():
            param.requires_grad = False
        self.fall_count = 0
        self.timeout_fail_count = 0
        self.last_critic_loss = None
        self.last_explore_critic_loss = None
        self.last_rnd_loss = None
        self.last_policy_loss = None
    def get_exploration_p(self, step, total_steps):
        # linear scheduling of exploration ratio
        start_val = 0
        end_val = 1
        if (len(self.modes) != 0):
            p = start_val + step * (end_val - start_val) / total_steps
        else: 
            p = 0.0
        return p
    def get_exploration_reward(self, state):
        pred_r = self.pred_exp_r_mlp(state)
        t = self.pred_exp_r_target(state)
        loss = (pred_r - t)*(pred_r-t)
        return loss.sum(-1)
    
         
    
    def select_embedding(self, p):
        
        embs = self.mode_explore_embedding.expand(self.num_envs, -1).clone()
        if (len(self.modes) != 0):
            num_refine_envs = round(p * self.num_envs)
            mode_ids = list(self.modes.keys())
            envs_per_mode = num_refine_envs // len(mode_ids)
            remainder = num_refine_envs % len(mode_ids)
            curr_insert_ind = 0
            for id in mode_ids:
                embs[curr_insert_ind:curr_insert_ind+envs_per_mode] = self.modes[id].embedding 
                curr_insert_ind+=envs_per_mode
            while(curr_insert_ind < num_refine_envs):
                id = mode_ids[curr_insert_ind % len(mode_ids)]
                embs[curr_insert_ind] = self.modes[id].embedding
                curr_insert_ind+=1
        return embs
        
    def explore_env(self, env, num_timesteps, rand, total_steps):
        """Collect environment transitions into a buffer.
        Args:
            env: The MuJoCo environment.
            num_timesteps (int): Total timesteps to collect per buffer.
            rand (bool): If True, enables random actions for warmup phase.
            total_steps (int): Total number of updates completed in the training session.
                Used to schedule exploration decay.
        """
        p = self.get_exploration_p(total_steps, params.total_train_steps)
        observation, info = env.reset()
        observation = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        t = 0
        success_streak = [0 for _ in range(self.num_envs)]
        episode_len = [0 for _ in range(self.num_envs)]
        while t < num_timesteps:

            if rand:
                action = env.action_space.sample()
                action_t = torch.as_tensor(action, dtype=torch.float32, device=self.device)
            else:
                # Schedules the rollout so that existing modes can be reinforced while new modes are still being discovered
                mode_emb = self.select_embedding(p)
                with torch.no_grad():
                    action_t = self.dp.get_actions(torch.cat([observation, mode_emb], dim=-1), self.num_envs)
                action = action_t.cpu().numpy()


            obs_prev = observation
            observation, reward, terminated, truncated, info = env.step(action)
            observation = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
            episode_over = np.logical_or(terminated, truncated)

            for i in range(self.num_envs):
                episode_len[i] += 1
                v_x = info["x_velocity"][i]
                if abs(v_x - self.target_vel[i]) <= 0.05:
                    success_streak[i] += 1
                with torch.no_grad():
                    raw_intrinsic_reward = self.get_exploration_reward(observation[i].unsqueeze(0)).squeeze(0)
                    self.rnd_reward_rms.update(raw_intrinsic_reward.cpu())
                    intrinsic_reward = self.rnd_reward_rms.normalize(raw_intrinsic_reward)
                ts = TransitionState(obs_prev[i], action_t[i], action_t[i], observation[i], reward[i]+intrinsic_reward, intrinsic_reward)
                self.trajectories[i].append(ts)
                if (episode_over[i]):
                    new_v_target = random.uniform(params.v_min, params.v_max)
                    self.target_vel[i] = new_v_target

                    if (terminated[i]):
                        self.fail_trajs.append((self.trajectories[i], self.traj_id))
                        self.fall_count += 1
                    elif (success_streak[i]/episode_len[i] > 0.5):
                        self.success_trajs.append((self.trajectories[i], self.traj_id))
                    else:
                        self.fail_trajs.append((self.trajectories[i], self.traj_id))
                        self.timeout_fail_count += 1
                    self.traj_id+=1
                    self.trajectories[i] = []
                    success_streak[i] = 0
                    episode_len[i] = 0
            t+=1
        self._evict_old_trajectories()
        return num_timesteps * self.num_envs

    def _evict_old_trajectories(self):
        total = len(self.success_trajs) + len(self.fail_trajs)
        if total <= params.memory_size:
            return
        to_evict = total - params.memory_size
        evicted_ids = set()
        success_evicted = False
        while to_evict > 0 and (self.success_trajs or self.fail_trajs):
            oldest_success_id = self.success_trajs[0][1] if self.success_trajs else None
            oldest_fail_id = self.fail_trajs[0][1] if self.fail_trajs else None
            if oldest_fail_id is not None and (oldest_success_id is None or oldest_fail_id < oldest_success_id):
                evicted_ids.add(self.fail_trajs.pop(0)[1])
            else:
                evicted_ids.add(self.success_trajs.pop(0)[1])
                success_evicted = True
            to_evict -= 1
        if evicted_ids:
            for mode in self.modes.values():
                mode.prev_cluster = [tid for tid in mode.prev_cluster if tid not in evicted_ids]
        if success_evicted:
            self.dist_matrix_cache = np.zeros((0, 0))  # force a full DTW recompute next process_trajs

    def new_mode(self, mode_id, prev_cluster, parent_mode=None, noise_scale=0.2):
        """create a new mode embedding and add to the modes dict"""
        if (parent_mode is None):
            vec = torch.randn(params.embed_dim, device=self.device) * noise_scale
        else:
            vec = parent_mode.embedding.detach().clone() + torch.randn(params.embed_dim, device=self.device) * noise_scale
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
            query = [ts.s.cpu().numpy() for ts in traj]
            best_idx, best_avg_dist = None, float('inf')
            for idx, cluster in enumerate(clusters):
                if not cluster:
                    continue
                sample_ids = random.sample(cluster, min(N, len(cluster)))
                dists = [dtw_ndim.distance(query, [ts.s.cpu().numpy() for ts in traj_by_id[sid]]) for sid in sample_ids]
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
        if N_new < 2:
            # not enough successful trajectories yet to form any clusters
            return None
        dist_matrix = np.zeros((N_new,N_new))
        if (N_old > 0):
            dist_matrix[:N_old, :N_old] = self.dist_matrix_cache
        for i in range(N_old, N_new):
            for j in range(N_new):
                if (i == j): continue
                states_i = [ts.s.cpu().numpy() for ts in self.success_trajs[i][0]]
                states_j = [ts.s.cpu().numpy() for ts in self.success_trajs[j][0]]
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
                    a_target = ts.a.clone()
                    for _ in range(params.action_update_times):
                        a_target = a_target.detach().requires_grad_(True)
                        qmin = critic.getqmin(ts.s, a_target)
                        a_grad = torch.autograd.grad(outputs=qmin, inputs=a_target)[0]
                        a_target = a_target + self.lr * a_grad
                    ts.a_target = a_target.detach()

        return dist_matrix
    def _mode_transitions(self, mode_id):
        """Flatten every transition belonging to trajectories currently assigned to mode_id."""
        transitions = []
        if (mode_id == -1):
            for traj, _ in self.success_trajs + self.fail_trajs:
                transitions.extend(traj)
            return transitions
        traj_by_id = {tid: traj for traj, tid in self.success_trajs + self.fail_trajs}
        for tid in self.modes[mode_id].prev_cluster:
            transitions.extend(traj_by_id[tid])
        return transitions

    def build_batch(self, batch_size=params.batch_size):
        """Returns a list of (sample, mode_embedding) pairs - each sample paired with
        the embedding of whichever mode it was drawn from, since a single multimodal
        batch mixes samples from different modes and the diffusion policy needs to
        know which mode's behavior each sample represents."""
        if not self.modes:
            pool = self._mode_transitions(-1)
            if not pool:
                return []
            samples = random.choices(pool, k=batch_size)
            return [(s, self.mode_explore_embedding) for s in samples]
        size_per_mode = batch_size // len(self.modes)
        batch = []
        for m_id in self.modes:
            pool = self._mode_transitions(m_id)
            if not pool:
                continue
            samples = random.choices(pool, k=size_per_mode)
            emb = self.modes[m_id].embedding
            batch += [(s, emb) for s in samples]
        return batch
    def update_critic(self):
        critic_losses = []
        for mode_id in self.modes:
            pool = self._mode_transitions(mode_id)
            if not pool:
                continue
            training_batch = random.choices(pool, k=params.batch_size)
            state = torch.stack([sample.s for sample in training_batch])
            action = torch.stack([sample.a for sample in training_batch])
            next_states = torch.stack([sample.s_next for sample in training_batch])
            rewards = torch.stack([sample.r for sample in training_batch])

            critic = self.Q_functions[mode_id]
            critic.q1.optimizer.zero_grad()
            critic.q2.optimizer.zero_grad()
            loss = critic.get_loss(state, action, next_states, rewards, self.modes[mode_id].embedding)
            loss.backward()
            critic_losses.append(loss.item())

            critic.update()
            critic.update_buffered(self.dp)
        self.last_critic_loss = sum(critic_losses) / len(critic_losses) if critic_losses else None

        # also update the Q_explore
        pool = self._mode_transitions(-1)
        if not pool:
            return
        training_batch = random.choices(pool, k=params.batch_size)
        state = torch.stack([sample.s for sample in training_batch])
        action = torch.stack([sample.a for sample in training_batch])
        next_states = torch.stack([sample.s_next for sample in training_batch])
        rewards = torch.stack([sample.r_intrinsic for sample in training_batch])
        critic = self.q_explore
        critic.q1.optimizer.zero_grad()
        critic.q2.optimizer.zero_grad()
        loss = critic.get_loss(state, action, next_states, rewards, self.mode_explore_embedding)
        loss.backward()
        self.last_explore_critic_loss = loss.item()
        critic.update()
        critic.update_buffered(self.dp)

        training_batch = random.choices(pool, k=params.batch_size)
        state = torch.stack([sample.s for sample in training_batch])
        self.exp_r_opt.zero_grad()
        loss = self.get_exploration_reward(state).mean()
        self.last_rnd_loss = loss.item()
        loss.backward()
        self.exp_r_opt.step()

    def update_policy(self):
        """Update diffusion policy weights indirectly using target action and behavorial cloning objective """
        training_batch = self.build_batch()
        if not training_batch:
            return
        self.dp.optimizer.zero_grad()
        loss = self.dp.get_loss(training_batch)
        loss.backward()
        self.last_policy_loss = loss.item()
        self.dp.update()
    

    