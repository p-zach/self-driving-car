# Author: Porter Zach
# Python 3.9

import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using " + device)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.stack = nn.Sequential(
            # vel, dir, 5 rays
            nn.Linear(9, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.ReLU()
        )

    def forward(self, x):
        y = self.stack(x)
        return y