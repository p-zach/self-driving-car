# Author: Porter Zach
# Python 3.9

import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.stack = nn.Sequential(
            # vel, lookdir, 5 rays
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh()
            # outputs forward accel and turn amount in range (-1, 1)
        )

    def forward(self, x):
        y = self.stack(x)
        return y