import torch.nn as nn

# --- dimensions (still placeholders — fill in from your actual Ant-v5 obs/action spaces) ---
transition_dim = 0
embed_dim = 0
action_dim = 0
state_dim = 0

# --- diffusion policy ---
diffusion_steps = 5   # cfg.diffusion.diffusion_iter; paper found 5 sufficient (Fig. 7b)
beta = 0.0             # placeholder; DiffusionPolicy currently hardcodes its own beta schedule

# --- critic / distributional RL ---
v_min = -4
v_max = 4

# --- rollout / training schedule ---
warm_up = 500              # cfg.algo.warm_up
horizon_len = 1             # cfg.algo.horizon_len
update_times = 8            # cfg.algo.update_times
batch_size = 4096           # cfg.algo.batch_size
eval_freq = 100             # cfg.eval_freq
log_freq = 2                 # cfg.log_freq (no logging wired up yet)
total_train_steps = 50000  # cfg.max_step — paper uses ~3M for AntMaze-v1; cfg's own fallback default is 4M, confirm which fits Ant-v5
memory_size = 2000          # cfg.algo.memory_size — NOT enforced yet; success_trajs/fail_trajs grow unbounded

# --- eval (no eval loop exists yet) ---
eval_num_envs = 20

# --- DIPO-style action-gradient refinement ---
action_update_times = 20   # cfg.diffusion.update_times — process_trajs currently only does 1 ascent step, not a K-loop