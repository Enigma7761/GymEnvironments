import torch
from torch import nn
from torch.nn import functional as F
import random
import numpy as np
from collections import deque

class PPO:
    def __init__(self, Actor, Value, horizon=128, lr=2.5e-4, gamma=0.99, gae_param=0.95, clipping_param=0.1, steps=32, VF=1,
                 entropy=0.01):
        self.policy = Actor()
        self.value = Value()
        self.horizon = horizon
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_param = gae_param
        self.clipping_param = clipping_param
        self.steps = 32
        self.VF = VF
        self.entropy = entropy

    def choose_action(self, state):

    def step(self):

    def compute_advantage(self, state, reward, next_state, done):

