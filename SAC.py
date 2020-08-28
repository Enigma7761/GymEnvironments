import torch
from torch import nn
from torch.nn import functional as F
import copy
import random
import numpy as np
from collections import deque

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DDPG:
    def __init__(self, Value, Q_func, Policy, action_space, lr=3e-4, buffer_size=100000,
                 gamma=0.99, batch_size=100, param_noise=0.1, max_distance=0.2, tau=0.005):
        self.value = Value().to(device)
        self.policy = Policy().to(device)
        self.value_target = Value().to(device)
        self.q = Q_func().to(device)
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr)
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr)
        self.replay = deque(maxlen=buffer_size)
        self.gamma = gamma
        self.batch_size=batch_size
        self.action_size = action_space.shape[0]
        self.high = action_space.high
        self.low = action_space.low
        for target_param, critic_param in zip(self.value_target.parameters(), self.value.parameters()):
            target_param.data.copy_(critic_param.data)

        self.param_noise = param_noise
        self.max_distance = max_distance
        self.tau = tau

    def soft_update(self):
        for target_param, critic_param in zip(self.value_target.parameters(), self.value.parameters()):
            target_param.data.copy_(self.tau * critic_param.data + (1 - self.tau) * target_param.data)

    def store(self, state, action, reward, next_state, done):
        self.replay.append((state, action, reward, next_state, done))

    def train(self):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        episodes = self.sample()
        if len(episodes) < self.batch_size:
            return
        for episode in episodes:
            states.append(episode[0])
            actions.append(episode[1])
            rewards.append(episode[2])
            next_states.append(episode[3])
            dones.append(0 if episode[4] else 1)
        logprobs = []
        for action, state in zip(actions, states):
            logprobs.append(self.policy(state).detach().numpy()[action])
        logprobs = torch.tensor(logprobs, dtype=torch.float32).to(device)
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device).reshape((self.batch_size, 1)).to(device)

        loss_value = F.mse_loss(self.value(states), (self.q(states, actions).detach() - logprobs))
        self.value_optim.zero_grad()
        loss_value.backward()
        self.value_optim.step()
        q_target = rewards + self.gamma*dones*self.value_target(next_states)
        loss_q = F.mse_loss(self.q(states, actions), q_target.detach())
        self.q_optim.zero_grad()
        loss_q.backward()
        self.q_optim.step()
        policy_loss = logprobs - 

    def sample(self):
        if len(self.replay) > self.batch_size:
            return random.sample(self.replay, self.batch_size)

    def choose_action(self, state):
        state = torch.tensor(state.copy(), dtype=torch.float32).to(device)
        self.policy.eval()
        with torch.no_grad():
            action = self.policy(state).cpu().detach().numpy()
        self.policy.train()
        action = np.clip(action, self.low, self.high)
        return action

    def apply_param_noise(self):
        with torch.no_grad():
            for param in self.policy.parameters():
                param.add_(torch.randn(param.size()).to(device) * self.param_noise)

    def adapt_noise(self, distance):
        if distance > self.max_distance:
            self.param_noise /= 1.01
        else:
            self.param_noise *= 1.01
