"""Shared utilities: device, seeding, model definition, flat-parameter helpers.

Reused near-verbatim from ../DBT-CFL-PTT/experiments/common.py (PCTA_Experimental_Methodology.md
B.3) -- same proven model/training pipeline, no redesign needed.
"""
import torch
import torch.nn as nn
import numpy as np
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SmallCNN(nn.Module):
    """Small CNN used for both CIFAR-10 (3x32x32) and Fashion-MNIST (1x28x28)."""

    def __init__(self, in_channels, num_classes, input_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        out_size = input_size // 4
        self.fc1 = nn.Linear(32 * out_size * out_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def get_flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()]).clone()


def set_flat_params(model, flat):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx:idx + n].view_as(p))
        idx += n


def num_params(model):
    return sum(p.numel() for p in model.parameters())
