# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules.actor_critic import ActorCritic

class ConstraintActorCritic(ActorCritic):
    """ConstraintActorCritic extends ActorCritic with a cost critic for constrained reinforcement learning.
    
    This class adds a separate cost value function that estimates the expected cost/penalty,
    which is essential for algorithms like PPO-Lagrangian that need to handle constraints.
    
    The architecture includes:
    - Actor: Policy network that outputs action distributions
    - Critic: Value function for rewards (inherited from ActorCritic)
    - Cost Critic: Value function for costs/penalties (new addition)
    """

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        cost_critic_hidden_dims=None,  # If None, uses same as critic_hidden_dims
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        """Initialize ConstraintActorCritic.
        
        Args:
            num_actor_obs: Number of actor observations
            num_critic_obs: Number of critic observations  
            num_actions: Number of actions
            actor_hidden_dims: Hidden layer dimensions for actor network
            critic_hidden_dims: Hidden layer dimensions for critic network
            cost_critic_hidden_dims: Hidden layer dimensions for cost critic network.
                                   If None, uses same as critic_hidden_dims
            activation: Activation function
            init_noise_std: Initial noise standard deviation
            noise_std_type: Type of noise standard deviation ('scalar' or 'log')
            **kwargs: Additional arguments passed to parent class
        """
        # Initialize parent ActorCritic class
        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )

        # Use same hidden dims as critic if not specified
        if cost_critic_hidden_dims is None:
            cost_critic_hidden_dims = critic_hidden_dims.copy()

        # Build cost critic network
        activation_fn = resolve_nn_activation(activation)
        mlp_input_dim_c = num_critic_obs
        
        cost_critic_layers = []
        cost_critic_layers.append(nn.Linear(mlp_input_dim_c, cost_critic_hidden_dims[0]))
        cost_critic_layers.append(activation_fn)
        
        for layer_index in range(len(cost_critic_hidden_dims)):
            if layer_index == len(cost_critic_hidden_dims) - 1:
                cost_critic_layers.append(nn.Linear(cost_critic_hidden_dims[layer_index], 1))
            else:
                cost_critic_layers.append(nn.Linear(cost_critic_hidden_dims[layer_index], cost_critic_hidden_dims[layer_index + 1]))
                cost_critic_layers.append(activation_fn)
        
        self.cost_critic = nn.Sequential(*cost_critic_layers)
        
        print(f"Cost Critic MLP: {self.cost_critic}")

    def evaluate_cost(self, critic_observations, **kwargs) -> torch.Tensor:
        """Evaluate the cost value function.
        
        Args:
            critic_observations: Critic observations tensor
            **kwargs: Additional arguments (for compatibility with recurrent networks)
            
        Returns:
            Cost values tensor
        """
        cost_value = self.cost_critic(critic_observations)
        return cost_value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the safe actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """
        # Check if the state_dict contains cost_critic parameters
        has_cost_critic = any(key.startswith('cost_critic') for key in state_dict.keys())
        
        if not has_cost_critic:
            print("Warning: Loading state_dict without cost_critic parameters. "
                  "Cost critic will be randomly initialized.")
            # Remove cost_critic from the model's state_dict for loading
            model_state_dict = self.state_dict()
            filtered_state_dict = {k: v for k, v in state_dict.items() 
                                 if not k.startswith('cost_critic')}
            model_state_dict.update(filtered_state_dict)
            # Load the updated state_dict
            self.load_state_dict(model_state_dict, strict=False)
        else:
            # Load all parameters including cost_critic
            super().load_state_dict(state_dict, strict=strict)
        
        return True
