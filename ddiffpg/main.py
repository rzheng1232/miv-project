import gymnasium as gym
import torch
import time
from ddiffpg import ddiffpg
from sim_wrapper import AntGaitWrapper
import params as params

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()

def main():
    env = gym.make_vec("Ant-v5", num_envs=256, vectorization_mode="sync", wrappers=[AntGaitWrapper])
    params.state_dim = env.single_observation_space.shape[0]
    params.action_dim = env.single_action_space.shape[0]
    params.embed_dim = 5  # mode embedding dim (Table 1)
    t_emb_size = 256       # matches the DiffusionNet(t_emb_size=256) constructions in ddiffpg.py
    params.transition_dim = t_emb_size + params.state_dim + params.action_dim
    m = ddiffpg(num_envs=256, lr=0.1)
    print(m.device)
    total_num_steps = 0
    # warmup
    m.explore_env(env, num_timesteps=params.warm_up, rand=True, total_steps=total_num_steps)
    m.process_trajs()
    iteration = 0
    while True:
        if iteration % params.eval_freq == 0:
            m.process_trajs()
            n_success = len(m.success_trajs)
            n_fail = len(m.fail_trajs)
            n_total = n_success + n_fail
            buffer_success_rate = n_success / n_total if n_total else float('nan')
            n_lifetime = m.success_count + m.fall_count + m.timeout_fail_count
            lifetime_success_rate = m.success_count / n_lifetime if n_lifetime else float('nan')
            all_trajs = m.success_trajs + m.fail_trajs
            avg_ep_len = sum(len(t) for t, _ in all_trajs) / len(all_trajs) if all_trajs else float('nan')
            mode_sizes = {mid: len(mode.prev_cluster) for mid, mode in m.modes.items()}
            p = m.get_exploration_p(total_num_steps, params.total_train_steps)
            avg_velocity = m.get_avg_velocity_and_reset()
            avg_dist = m.get_avg_dist_and_reset()
            print(
                f"[iter {iteration}] steps={total_num_steps} modes={len(m.modes)} mode_sizes={mode_sizes}\n"
                f"  buffer: success_rate={buffer_success_rate:.3f} n={n_total} (capped at memory_size={params.memory_size})\n"
                f"  lifetime: success_rate={lifetime_success_rate:.3f} n={n_lifetime} "
                f"(successes={m.success_count} falls={m.fall_count} timeouts={m.timeout_fail_count})\n"
                f"  avg_ep_len={avg_ep_len:.1f} avg_x_velocity={avg_velocity:.3f} avg_dist_to_goal={avg_dist:.3f} explore_p={p:.3f}\n"
                f"  critic_loss={m.last_critic_loss} explore_critic_loss={m.last_explore_critic_loss} "
                f"rnd_loss={m.last_rnd_loss} policy_loss={m.last_policy_loss}\n"
            )
        # sync(); t0 = time.perf_counter()
        total_num_steps += m.explore_env(env, num_timesteps=params.horizon_len, rand=False, total_steps=total_num_steps)
        # sync(); t1 = time.perf_counter()
        for i in range(params.update_times):
            m.update_critic()
            m.update_policy()
        # sync(); t2 = time.perf_counter()
        # print(f"[iter {iteration}] explore_env={t1-t0:.3f}s updates={t2-t1:.3f}s")
        if (total_num_steps > params.total_train_steps):
            break
        iteration+=1


main()