import torch
from torch import nn
from torch.nn import functional as F
from ReplayBuffer import ReplayBuffer, PrioritizedReplayBuffer
import random
import numpy as np
from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQN:
    def __init__(self, Model, minibatch_size=64, replay_memory_size=1000000, gamma=0.99, learning_rate=5e-4,
                 tau=1e-4, param_noise=0.1, max_distance=0.2, alpha=0.5, beta=0.5):
        self.minibatch_size = minibatch_size
        self.replay_memory_size = replay_memory_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.tau = tau
        self.value = Model().to(device)
        self.target1 = Model().to(device)
        self.target1.eval()
        self.copy_weights()
        self.replay = PrioritizedReplayBuffer(replay_memory_size, minibatch_size, alpha)
        self.copy_weights()
        self.param_noise = param_noise
        self.max_distance = max_distance
        self.optimizer = torch.optim.Adam(self.value.parameters(), lr=self.learning_rate)
        self.beta = beta

    def copy_weights(self):
        for target_param, value_param in zip(self.target1.parameters(), self.value.parameters()):
            target_param.data.copy_(value_param.data)
        self.target1.eval()

    def soft_update(self):
        for target_param, value_param in zip(self.target1.parameters(), self.value.parameters()):
            target_param.data.copy_(value_param.data * self.tau + target_param.data * (1 - self.tau))

    def choose_action(self, state):
        self.value.eval()
        state = torch.from_numpy(state).to(device, torch.float32)
        action = torch.argmax(self.value(state), dim=1).detach().cpu().numpy()
        self.value.train()
        return action

    def store(self, state, action, reward, next_state, done):
        self.replay.store(state, action, reward, next_state, done)

    def train(self):
        if len(self.replay) < self.minibatch_size:
            return
        states, actions, rewards, next_states, dones, probs, indices = self.replay.sample()
        weights = (probs * len(self.replay)) ** (-self.beta)
        weights = weights / np.max(weights)
        weights = torch.tensor(weights, device=device, dtype=torch.float32)

        n = rewards.shape[1]

        states = torch.from_numpy(states).to(device, torch.float32).unsqueeze(1)
        next_states = torch.from_numpy(next_states).to(device, torch.float32).unsqueeze(1)
        rewards = torch.from_numpy(rewards).to(device, torch.float32)
        gammas = torch.tensor([self.gamma ** i for i in range(n)], dtype=torch.float32, device=device)
        dones = torch.from_numpy(dones).to(device, torch.float32)
        actions = torch.from_numpy(actions).to(device, torch.long).unsqueeze(1)

        target = torch.sum(rewards * gammas, dim=1) + (self.gamma ** n) * self.target1(next_states).detach().gather(1,
                       torch.argmax(self.value(next_states).detach(), dim=1).unsqueeze(1)).squeeze(1) * (1 - dones)

        target = target.unsqueeze(1)
        expected = self.value(states).gather(1, actions)
        self.optimizer.zero_grad()
        loss = (weights * ((target - expected) ** 2).squeeze(1)).mean()
        loss.backward()
        self.optimizer.step()
        # loss = F.mse_loss(target, expected)
        updated_priorities = torch.abs(target - expected).detach().cpu().numpy() + 0.001
        self.replay.store_priorities(indices, updated_priorities.squeeze(1))
        temp = loss.detach().cpu().item()

        return temp
