import torch
from torch import nn
from torch.nn import functional as F
import random
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.pi = (torch.acos(torch.zeros(1)) * 2).to(device)
ENTROPY_BETA = 1e-4
# device = ('cpu')

class A2C:
    def __init__(self, Actor, Critic, action_space):
        self.actor = Actor().to(device)
        self.critic = Critic().to(device)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr = 0.001)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=0.0001)
        self.gamma = 0.99
        self.epsilon = 0
        self.epsilon_min = 0
        self.action_dim = action_space.shape[0]
        self.high = action_space.high
        self.low = action_space.low

    def train(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32, device=device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
        action = torch.tensor(action, dtype=torch.float32, device=device)
        reward = torch.tensor(reward, dtype=torch.float32, device=device)

        done = torch.tensor(done, dtype=torch.float32, device=device).unsqueeze(1)

        val_loss = F.mse_loss(reward.unsqueeze(1) + self.gamma*done*self.actor(next_state)[2], self.actor(state)[2])
        # self.critic_optim.zero_grad()
        # val_loss.backward()
        # self.critic_optim.step()
        TD = reward + self.gamma*done*self.actor(next_state)[2] - self.actor(state)[2]

        mu, var, _ = self.actor(state)
        var += self.epsilon
        a = (self.high - self.low)/2
        a = torch.tensor(a, dtype=torch.float32, device=device)
        log_prob = (-((action - mu) ** 2) / (2*var.clamp(min=1e-3)) - torch.log(torch.sqrt(2 * torch.pi * var.clamp(min=1e-3)))) * TD.detach()
        entropy_loss = (-(torch.log(2*torch.pi*var) + 1)/2) * ENTROPY_BETA
        loss = -(val_loss + log_prob + entropy_loss).mean()
        # print("Prob", log_prob)

        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=device)
        # print(state)
        # print(self.actor(state)[0])
        mu, var, _ = self.actor(state)
        mu = mu.data.cpu().numpy()
        std = torch.sqrt(var + self.epsilon).data.cpu().numpy()
        action = np.random.normal(mu, std)
        np.clip(action, self.low, self.high)
        return action

    def decrement_epsilon(self):
        self.epsilon = max(self.epsilon*0.95, self.epsilon_min)