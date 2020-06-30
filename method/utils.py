import moviepy.editor as mpy
import os.path as osp
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

def add_mag_stats(inferred_z):
    mags = torch.sum(inferred_z ** 2.0, dim=-1).sqrt()
    return {
        'alg_add_z_mag_min' : mags.min().cpu().item(),
        'alg_add_z_mag_max' : mags.max().cpu().item(),
        'alg_add_z_mag_mean': mags.mean().cpu().item()
    }


class SourceTargetDataset(Dataset):
    def __init__(self, x, y, is_cuda=False):
        self.x = x
        self.y = y
        if is_cuda:
            self.x = self.x.cuda()
            self.y = self.y.cuda()
        else:
            self.x = self.x.cpu()
            self.y = self.y.cpu()

        assert self.x.shape[0] == self.y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]

class EffectDiscrim(nn.Module):
    def __init__(self, z_dim, a_dim, hidden_dim = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + a_dim,  hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, z_emb, a_emb):
        inputs = torch.cat([z_emb, a_emb], dim=-1)
        return self.net(inputs)


class EffectGen(nn.Module):
    def __init__(self, z_dim, a_dim, noise_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(noise_dim + a_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, z_dim))

    def forward(self, noise, a_emb):
        inputs = torch.cat([noise, a_emb], dim=-1)
        return self.net(inputs)

def one_hotify(indices, num_classes):
    one_hot = torch.zeros((indices.shape[0], num_classes))
    if indices.is_cuda:
        one_hot = one_hot.cuda()
    one_hot.scatter_(1, indices, 1)
    return one_hot

class OneHotDist(D.categorical.Categorical):
    def __init__(self, logits):
        super().__init__(logits=logits)
        self.num_classes = logits.shape[-1]

    def sample(self):
        indices = super().sample()

        return one_hotify(indices, self.num_classes)

    def log_prob(self, value):
        indices = value.max(-1)[1]
        return super().log_prob(indices)


# Building block for convolutional encoder with same padding
class Conv2d3x3(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(Conv2d3x3, self).__init__()
        stride = 2 if downsample else 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              padding=1, stride=stride)

    def forward(self, x):
        return self.conv(x)


