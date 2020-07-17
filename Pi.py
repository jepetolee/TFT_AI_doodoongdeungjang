import torch as T
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np


class G_net(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(G_net, self).__init__()
        self.input_dims = input_dims
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.Device = T.device('cuda:0' if T.cuda.is_available() else 'cpu:0')
        self.to(self.Device)

    def forward(self, observation):
        state = T.tensor(observation).to(self.Device)
        x = f.relu(self.fc1(state))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        return x


class agent(object):
    def __init__(self, a, b, input_dims, g=0.99, l1_size=256, l2_size=256, n_actions=2):
        self.g = g
        self.log = None
        self.actor = G_net(a, input_dims, l1_size, l2_size, n_actions)
        self.critic = G_net(b, input_dims, l1_size, l2_size, n_actions)
