import torch
from torch import nn
from torch.nn import functional as F
from ReplayBuffer import PrioritizedReplayBuffer
import random
import numpy as np
from collections import deque

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class TD3:
    def __init__(self, Actor, Critic, action_space, replay_size=1000000, critic_lr=1e-3, training=True,
                 actor_lr=1e-3, gamma=0.99, batch_size=100, tau=5e-3, update_freq=2, alpha=0.5, beta=0.5,
                 noise_std=0.1, noise_clip=0.5, seed=0):
        torch.manual_seed(0)
        np.random.seed(seed)
        self.critic1 = Critic().to(device)
        self.critic2 = Critic().to(device)
        self.actor = Actor().to(device)
        self.critic_target1 = Critic().to(device)
        self.critic_target2 = Critic().to(device)
        self.actor_target = Actor().to(device)
        self.critic_optim1 = torch.optim.Adam(self.critic1.parameters(), critic_lr)
        self.critic_optim2 = torch.optim.Adam(self.critic2.parameters(), critic_lr)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), actor_lr)
        self.replay = deque(maxlen=replay_size)
        self.gamma = gamma
        self.batch_size = batch_size
        self.action_size = action_space.shape[0]
        self.high = action_space.high
        self.low = action_space.low
        self.replay = PrioritizedReplayBuffer(replay_size, batch_size, alpha)
        for target_param, critic_param in zip(self.critic_target1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(critic_param.data)
        for target_param, critic_param in zip(self.critic_target2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(critic_param.data)
        for target_param, actr_param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(actr_param.data)

        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.beta = beta
        self.update_freq = update_freq
        self.tau = tau
        self.training = training

    def soft_update(self):
        for target_param, critic_param in zip(self.critic_target1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * critic_param.data + (1 - self.tau) * target_param.data)
        for target_param, critic_param in zip(self.critic_target2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * critic_param.data + (1 - self.tau) * target_param.data)
        for target_param, actr_param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * actr_param.data + (1 - self.tau) * target_param.data)

    def store(self, state, action, reward, next_state, done):
        self.replay.store(state, action, reward, next_state, done)

    def train(self, t):
        if len(self.replay) < self.batch_size:
            return
        states, actions, rewards, next_states, dones, probs, indices = self.replay.sample()
        weights = (probs * len(self.replay)) ** (-self.beta)
        weights = weights / np.max(weights)
        weights = torch.tensor(weights, device=device, dtype=torch.float32)

        n = rewards.shape[1]

        states = torch.tensor(states, device=device, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, device=device, dtype=torch.float32).unsqueeze(1)
        rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
        gammas = torch.tensor([self.gamma ** i for i in range(n)], dtype=torch.float32, device=device)
        dones = torch.tensor(dones, device=device, dtype=torch.float32)
        actions = torch.tensor(actions, device=device, dtype=torch.long).unsqueeze(1)
        target_action = self.actor_target(next_states)
        action_noise = torch.normal(mean=torch.zeros(size=[self.batch_size, self.action_size]),
                                    std=torch.ones(size=[self.batch_size, self.action_size]) * self.noise_std) \
            .clamp(-self.noise_clip, self.noise_clip).to(device)
        target_action += action_noise
        target_action = target_action.detach().cpu().numpy()
        target_action = torch.from_numpy(np.clip(target_action, self.low, self.high)).to(device)
        target = rewards.unsqueeze(1) + (1 - dones) * gammas * torch.min(
            self.critic_target1(next_states, target_action.detach()),
            self.critic_target2(next_states, target_action.detach()))
        loss1 = weights * (target - self.critic1.forward(states, actions))**2
        td = torch.abs(target - self.critic1.forward(states, actions)).detach().cpu().numpy() + 0.001
        self.replay.store_priorities(indices, td.squeeze(1))
        self.critic_optim1.zero_grad()
        loss1.backward(retain_graph=True)
        self.critic_optim1.step()
        loss2 = weights * (target - self.critic2.forward(states, actions))**2
        self.critic_optim2.zero_grad()
        loss2.backward()
        self.critic_optim2.step()
        if t % self.update_freq:
            policy_loss = -weights * self.critic1.forward(states, self.actor.forward(states)).mean()
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
        action += np.random.normal(0, 0.1, self.action_size)
        action = np.clip(action, self.low, self.high)
        return action

