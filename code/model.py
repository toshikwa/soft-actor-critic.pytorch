import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MAX = 2
LOG_STD_MIN = -20
epsilon = 1e-6


def weights_init_xavier(m):
    """ Initialize weights with Xavier's initializer. """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def weights_init_he(m):
    """ Initialize weights with He's initializer. """
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):
    """ Pairs of two Q-networks. """

    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1
        self.linear1 = nn.Linear(
            num_inputs + num_actions, hidden_dim).apply(weights_init_he)
        self.linear2 = nn.Linear(
            hidden_dim, hidden_dim).apply(weights_init_he)
        self.linear3 = nn.Linear(
            hidden_dim, 1).apply(weights_init_xavier)

        # Q2
        self.linear4 = nn.Linear(
            num_inputs + num_actions, hidden_dim).apply(weights_init_he)
        self.linear5 = nn.Linear(
            hidden_dim, hidden_dim).apply(weights_init_he)
        self.linear6 = nn.Linear(
            hidden_dim, 1).apply(weights_init_xavier)

    def forward(self, state, action):
        # state-action pair
        xu = torch.cat([state, action], 1)

        # pass forward
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class GaussianPolicy(nn.Module):
    """ Gaussian policy with reparameterization tricks. """

    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        # dense layers
        self.linear1 = nn.Linear(num_inputs, hidden_dim).apply(weights_init_he)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim).apply(weights_init_he)

        # last layer to the mean of gaussian
        self.mean_linear = nn.Linear(
            hidden_dim, num_actions).apply(weights_init_xavier)
        # last layer to the log(std) of gaussian
        self.log_std_linear = nn.Linear(
            hidden_dim, num_actions).apply(weights_init_xavier)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = (action_space.high - action_space.low) / 2.
            self.action_bias = (action_space.high + action_space.low) / 2.

    def forward(self, state):
        # pass forward
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        # mean of gaussian
        mean = self.mean_linear(x)
        # log(std) of gaussian
        log_std = self.log_std_linear(x)
        # clip the log(std)
        log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)

        return mean, log_std

    def sample(self, state):
        # mean, log(std)
        mean, log_std = self.forward(state)
        # std
        std = log_std.exp()
        # gaussian distribution
        normal = Normal(mean, std)
        # sample with reparameterization tricks
        x_t = normal.rsample()
        # action
        action = torch.tanh(x_t) * self.action_scale + self.action_bias
        # log likelihood
        log_prob = normal.log_prob(x_t)\
            - torch.log(self.action_scale * (1 - action.pow(2)) + epsilon)
        # sum through all actions
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, torch.tanh(mean)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
