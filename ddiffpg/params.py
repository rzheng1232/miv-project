import torch.nn as nn
transition_dim = 0
embed_dim = 0
action_dim = 0
state_dim = 0
diffusion_steps = 5  # paper found 5 sufficient (Fig. 7b)
beta = 0.0  # placeholder; DiffusionPolicy currently hardcodes its own beta schedule