import jax.numpy as jnp
from jax import grad, jit, jacobian, random
from functools import partial

def F(x, a, theta):
    return 1 / (1 + jnp.exp(-a * (x - theta))) - 1 / (1 + jnp.exp(a * theta))
def WC_rhs(r, W_scaled, I_ext, tau, a_sigmoid, theta):
    return -r / tau + F(W_scaled @ r + I_ext, a_sigmoid, theta)
WC_jac = jacobian(WC_rhs, argnums=0)

def _dynamics_matrices(is_exc, pos_init, a_dist, tau_E, tau_I, W):
    r_pos = jnp.asarray(pos_init)
    diff = r_pos[:, None, :] - r_pos[None, :, :]          # (N, N, 2)
    dist = jnp.linalg.norm(diff, axis=-1)                  # (N, N)
    N = r_pos.shape[0]
    pairwise_dist_factor_matrix = jnp.exp(-a_dist * dist) * (1 - jnp.eye(N))
    W_scaled = W * pairwise_dist_factor_matrix
    tau = jnp.where(is_exc, tau_E, tau_I)
    return W_scaled, tau
def simulate_wc_w_dist(init_state, is_exc, pos_init, I_ext, range_t, a_dist, a_sigmoid,theta, W, dt, tau_I, tau_E):
    """inputs:
        init_state: initial state vector (e.g., firing rates)
        is_exc: boolean array indicating excitability of each neuron
        pos_init: initial positions
        range_t: time range
        a_dist: distance decay parameter
        a_sigmoid: sigmoid steepness parameter
        W: connectivity matrix (negative weights for inhibition, positive for excitation)
        dt: time step
        tau_I: time constant for inhibitory neurons
        tau_E: time constant for excitatory neurons
        theta: threshold for the sigmoid function
    """
    Lt = range_t.size
    r = jnp.zeros((Lt, len(init_state))).at[0].set(init_state)
    ext_arr = I_ext * jnp.ones((Lt, len(init_state)))
    W_scaled, tau = _dynamics_matrices(is_exc, pos_init, a_dist, tau_E, tau_I, W)
    for k in range(Lt - 1):
        dr = dt / tau * (-r[k] + F(W_scaled @ r[k] + ext_arr[k], a_sigmoid, theta))
        r = r.at[k + 1].set(r[k] + dr)

    return r
def newton_method(approx_steps, init_state, is_exc, pos_init, I_ext, a_dist, a_sigmoid,theta, W, tau_I, tau_E):

    r = jnp.asarray(init_state)
    W_scaled, tau = _dynamics_matrices(is_exc, pos_init, a_dist, tau_E, tau_I, W)

    for k in range(approx_steps):
        f =  WC_rhs(r, W_scaled, I_ext, tau, a_sigmoid, theta)
        j =  WC_jac(r, W_scaled, I_ext, tau, a_sigmoid, theta)
        # newton method is: r_t+1 = r_t - f / j = r_t+1 = r_t - f j^-1
        r = r - jnp.linalg.solve(j, f)
    final_residual = jnp.linalg.norm(WC_rhs(r, W_scaled, I_ext, tau, a_sigmoid, theta))
    # print(f"final residual: {final_residual}")
    return r, final_residual
def find_fixed_points(params, steps, num_samples, seed=0):
    is_exc = params["exc_map"]
    pos_init = params["2dmap"]
    I_ext = params["ext_input"]
    a_dist = params["distance_decay"]
    a_sigmoid = params["sigmoid_steepness"]
    theta = params["theta"]
    W = params["weights"]
    tau_I = params["tau_I"]
    tau_E = params["tau_E"]
    fixed_points_disovered = []
    key = random.PRNGKey(seed)
    for _ in range(num_samples):
        key, subkey = random.split(key)
        start_state = random.uniform(subkey, shape=(len(is_exc),))
        discovered_fixed_point, res = newton_method(steps, start_state, is_exc, pos_init, I_ext, a_dist, a_sigmoid, theta, W, tau_I, tau_E)
        if res < 1e-4 and not jnp.isnan(res):
            fixed_points_disovered.append(discovered_fixed_point)
    return fixed_points_disovered
def jacobian_at(r, params):
    is_exc = params["exc_map"]
    pos_init = params["2dmap"]
    I_ext = params["ext_input"]
    a_dist = params["distance_decay"]
    a_sigmoid = params["sigmoid_steepness"]
    theta = params["theta"]
    W = params["weights"]
    tau_I = params["tau_I"]
    tau_E = params["tau_E"]
    W_scaled, tau = _dynamics_matrices(is_exc, pos_init, a_dist, tau_E, tau_I, W)
    return WC_jac(r, W_scaled, I_ext, tau, a_sigmoid, theta)
def merge_fixed_points(fixed_points, tol=1e-3):
    clusters = []
    for fp in fixed_points:
        placed = False
        for cluster in clusters:
            if jnp.linalg.norm(fp - cluster[0]) < tol:
                cluster.append(fp)
                placed = True
                break
        if not placed:
            clusters.append([fp])
    return [jnp.mean(jnp.stack(c), axis=0) for c in clusters]