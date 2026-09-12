### Part 1: DDiffPG Implementation

**Background:** There is insufficient exploration to discover multiple modes, and the inherent greediness of standard RL objectives causes a policy to collapse onto just one mode even after finding several modes.
DDiffPG addresses the problem of limited exploration by using mode
clustering and a shifting behavioral-cloning target, calculated with the action
gradient, to learn multi-modal behaviors from scratch in sparse-reward
environments.

**The challenge:** Use DDiffPG to learn multi-modal behavior in the Gymnasium
Ant-v5 and MsPacman environments.

**My approach:**

- I rewrote the architecture largely from scratch, using the original DDiffPG
  codebase as a reference for code structure and specific parameter details.
- DDiffPG heavily relies on sparse reward and a binary success indicator, which
  is difficult in Ant-v5:
  - The goal in Ant-v5 is typically to learn movement and stable gaits on the
    3D ant robot. Unlike the original paper's AntMaze environment, there's no
    clear binary state for whether the policy has "succeeded."
  - **First attempt:** based on prior experience training quadrupeds, I had
    the ant try to maintain a specified stable velocity target, to see if
    multiple gait modes would emerge for traveling at the same velocity (then
    use cost-of-transport to check whether the discovered gaits were roughly
    equivalent in energy cost). Success meant maintaining velocity within a
    threshold of the target over a specified number of consecutive steps
    without falling. This failed because, before the algorithm discovers any
    successful trajectory, mode discovery relies solely on the intrinsic
    exploration reward (driven by state novelty, not any goal), and
    maintaining velocity across consecutive timesteps was too hard a task for
    that novelty-driven exploration to ever stumble into a successful mode.
  - **Second attempt:** I wrote a wrapper setting a target goal distance and velocity,
    closer to AntMaze's sparse indicator, with success meaning reaching the target
    distance from the starting state, with the velocity error (rather than
    velocity itself) injected as a reward term. This also failed to discover a successful mode in the testing I've done (my macbook was only able to finish ~1,000,000 timesteps in a reasonable timeframe), mainly because there was nothing driving the policy to maintain a forward heading until it reaches the goal; Although the AntMaze problem similar, the maze has walls that will guide the policy, even if it is randomly moving, to the goal state eventually. 
- At this point I was running out of time for further reward engineering on
  Ant, so I moved on to the next task.
- I did not get a chance to test the MsPacman environment, but here is my
  intended approach:
  - Initially, I expected to approach this problem with a straightforward
  drop-in of the DDiffPG algorithm into the MsPacman environment. However, I
  realized that DDiffPG's core mechanism relies on differentiating the critic
  (Q-function) with respect to the action to compute an action gradient, which
  is used to calculate a shifting behavioral-cloning target. Since MsPacman is
  a discrete action environment, there isn't an action gradient to compute in
  the same sense. My current plan is to swap the diffusion action head for a
  mode-conditioned categorical policy, while keeping the rest of the
  mode-discovery and mode-conditioning machinery. In this approach, the
  behavioral-cloning target would be a probability distribution over the
  discrete actions rather than a single action vector, likely derived by
  reweighting the target distribution toward higher-Q actions (e.g.
  exponentially, similar to soft Q-learning), rather than literal gradient
  ascent, though I did not have time to work out or validate the exact update
  rule.

### Part 2: Wilson-Cowan Rate-Based Neuron Model

**Background:** The Wilson-Cowan model (Wilson & Cowan, 1972) is a foundational
rate-based model of coupled excitatory (E) and inhibitory (I) neural
populations. It describes how each population's average firing rate evolves
over time as a function of recurrent connectivity, external input, and a
saturating sigmoid nonlinearity representing the population's collective
input-output response — making it a minimal model for studying how the
excitation/inhibition balance produces stable states, oscillations, and
pattern formation in neural circuits.

The goal of this task was to identify parameter regimes — varying excitatory
coupling strength, inhibitory coupling strength, and external input — that
produce stable activity, oscillatory dynamics, and spatially structured
activity.

**My approach:**

- I wrote my model in JAX, basing parts of the simulation code on the
  [Neuromatch Wilson-Cowan tutorial](https://compneuro.neuromatch.io/tutorials/W2D5_DynamicalSystems/student/W2D5_Tutorial2.html).
- To characterize the dynamical systems behavior, I identified the best method
  as analyzing the eigenvalues of the Jacobian at the system's fixed points.
  The rules for interpreting eigenvalues at a fixed point are:
  - `+real / 0 imag` = unstable node
  - `-real / 0 imag` = stable node
  - `+real / not 0 imag` = unstable spiral (repellor)
  - `-real / not 0 imag` = stable spiral (attractor)
  - `0 real / not 0 imag` = center (the marginal boundary case — the exact
    Hopf bifurcation transition between stable and unstable spirals)
- To solve for fixed points of a high-dimensional dynamical system, I use a
  multi-start Newton's method (`s_{t+1} = s_t - f/f'`, where `f' = J`),
  evenly sampling N points across the state space and running Newton's method
  from each to find local fixed points, then eliminating duplicates by
  clustering near-identical values and averaging.
- I use exponential distance decay to scale synaptic weights between neuron
  populations based on their geometric location on a flat 2D plane (positions
  randomly scattered): `W_scaled[i,j] = W_raw[i,j] * exp(-a_dist * d(i,j))`.
- I initialize an 80:20 ratio of excitatory to inhibitory neurons, matching
  the approximate E:I ratio found in the human brain, and set the inhibitory
  time constant (response speed) to half the excitatory time constant, also
  consistent with biological values.
- I swept three sets of parameters: inhibitory weights (all connections
  sourced from inhibitory neurons), excitatory weights (all connections
  sourced from excitatory neurons), and external excitatory and inhibitory
  input (swept jointly).
- Specific findings and observations are in the Python notebook (with graphs).
