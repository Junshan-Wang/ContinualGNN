import os
import sys
import random 

import torch
import torch.nn as nn



class SAGEConv(torch.nn.Module):

    def __init__(self, in_size, out_size):
        super(SAGEConv, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(torch.Tensor(self.in_size * 2, self.out_size))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, x, aggregate_x):
        combine_x = torch.cat([x, aggregate_x], dim = 1)
        return nn.functional.relu(torch.matmul(combine_x, self.weight))
