import torch
from torch import nn
from torch.nn import functional as F
import copy
import random
import numpy as np
from collections import deque

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DDPG:
    def __init__(self, Actor, Critic, action_space, replay_size=1000000, critic_lr=0.001,
                 actor_lr=0.0001, gamma=0.99, batch_size=64, param_noise=0.1, max_distance=0.2, tau=1e-3):
        self.critic = Critic().to(device)
        self.actor = Actor().to(device)
        self.critic_target = Critic().to(device)
        self.actor_target = Actor().to(device)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), critic_lr)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), actor_lr)
        self.replay = deque(maxlen=replay_size)
        self.gamma = gamma
        self.batch_size=batch_size
        self.action_size = action_space.shape[0]
        self.high = action_space.high
        self.low = action_space.low
        for target_param, critic_param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(critic_param.data)
        for target_param, actr_param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(actr_param.data)

        self.param_noise = param_noise
        self.max_distance = max_distance
        self.tau = tau

    def soft_update(self):
        for target_param, critic_param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * critic_param.data + (1 - self.tau) * target_param.data)
        for target_param, actr_param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * actr_param.data + (1 - self.tau) * target_param.data)

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
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        actions = actions.squeeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        target_action = self.actor_target(next_states)
        target = rewards.unsqueeze(1) + self.gamma * self.critic_target(next_states, target_action.detach())
        criterion = nn.MSELoss()
        loss = criterion(target, self.critic.forward(states, actions))
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()
        self.soft_update()

    def sample(self):
        if len(self.replay) > self.batch_size:
            return random.sample(self.replay, self.batch_size)
        else:
            return self.replay

    def choose_action(self, state):
        state = torch.tensor(state.copy(), dtype=torch.float32).to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().detach().numpy()
        self.actor.train()
        action = np.clip(action, self.low, self.high)
        return action

    def apply_param_noise(self):
        with torch.no_grad():
            for param in self.actor.parameters():
                param.add_(torch.randn(param.size()).to(device) * self.param_noise)

    def adapt_noise(self, distance):
        if distance > self.max_distance:
            self.param_noise /= 1.01
        else:
            self.param_noise *= 1.01
