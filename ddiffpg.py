class ddiffpg():
    def __init__(self):
        diffusion_buffer = []
        tot_updates = 0 # track progress of training


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
        """Update diffusion policy weights indirectly using target action and behavorial cloning objective"""

    
    
