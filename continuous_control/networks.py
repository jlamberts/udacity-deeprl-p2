import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    """Actor Critic network for continuous action spaces.  This network assumes that the output"""

    def __init__(
        self,
        state_size=33,
        action_size=4,
        hidden_layer_size=256,
        seed=42,
        batchnorm_inputs=True,
    ):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_layer_size (int): Size of the hidden layer
            seed (int): Random seed
            batchnorm_inputs (bool): if True, apply batch normalization to the inputs
                Per Lillicrap et al (2016) this can help training and generalization with different physical dimensions for inputs
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size

        self.batchnorm_layer = nn.BatchNorm1d(state_size) if batchnorm_inputs else None

        self.inputs_actor = nn.Linear(state_size, hidden_layer_size)
        self.action_means = nn.Linear(hidden_layer_size, action_size)
        self.action_stds = nn.Linear(hidden_layer_size, action_size)

        # log std vs directly calculating std seems to improve stability
        # idea from https://medium.com/deeplearningmadeeasy/advantage-actor-critic-continuous-case-implementation-f55ce5da6b4c
        self.log_std = nn.Parameter(torch.zeros(action_size))

    def forward(self, state):
        """Build a network that maps state -> policy, value"""
        if self.batchnorm_layer:
            state = self.batchnorm_layer(state)

        # calculate policy using actor network
        policy = self.inputs_actor(state)
        policy = F.relu(policy)

        # tanh will give us an output in the range (-1, 1)
        policy_mean = F.tanh(self.action_means(policy))

        # create one distribution per action
        policy_dist = torch.distributions.Normal(
            policy_mean, torch.clamp(self.log_std.exp(), 1e-3, 50)
        )

        return policy_dist


class CriticNetwork(nn.Module):
    """Actor Critic network for continuous action spaces.  This network assumes that the output"""

    def __init__(
        self,
        state_size=33,
        action_size=4,
        hidden_layer_size=256,
        seed=42,
        batchnorm_inputs=True,
    ):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_layer_size (int): Size of the hidden layer
            seed (int): Random seed
            batchnorm_inputs (bool): if True, apply batch normalization to the inputs
                Per Lillicrap et al (2016) this can help training and generalization with different physical dimensions for inputs
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size

        self.batchnorm_layer = nn.BatchNorm1d(state_size) if batchnorm_inputs else None

        # Using separate networks: https://datascience.stackexchange.com/questions/35814/confusion-about-neural-network-architecture-for-the-actor-critic-reinforcement-l
        self.inputs_critic = nn.Linear(state_size, hidden_layer_size)
        self.outputs_critic = nn.Linear(hidden_layer_size, 1)

    def forward(self, state):
        """Build a network that maps state -> policy, value"""
        if self.batchnorm_layer:
            state = self.batchnorm_layer(state)
        # calculate policy using actor network
        value = self.inputs_critic(state)
        value = F.relu(value)
        value = self.outputs_critic(value)
        return value