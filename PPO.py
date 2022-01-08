import torch
from torch import nn
from torch.nn import functional as F
import random
import numpy as np
from collections import deque
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()


class PPO:
    def __init__(self, ActorCritic, n_envs, RND=None, horizon=128, lr=2.5e-4, gamma=0.99, i_gamma=0.99, gae_param=0.95,
                 VF=1, entropy=0.01, epochs=10, clip=0.2, batch_size=64):
        self.n_envs = n_envs
        if RND is not None:
            self.predictor = RND().to(device)
            self.network = RND().to(device)
            self.predictor_optim = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        else:
            self.predictor = None
            self.network = None
        self.model = ActorCritic().to(device)
        self.old_model = ActorCritic().to(device)
        self.horizon = horizon
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_param = gae_param
        self.VF = VF
        self.entropy = entropy
        self.epochs = epochs
        self.clip = clip
        self.advantage_matrix = np.array(
            [[(self.gamma * self.gae_param) ** (i - j + 1) if i - j + 1 > 0 else 0 for j in range(horizon)] for i in
             range(horizon)])
        self.counter = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.intrinsic_rewards = []
        self.dones = []
        self.batch_size = batch_size
        self.update_count = 0
        self.sumx = 0
        self.sumx2 = 0
        self.num = 0
        self.i_gamma = i_gamma

    def choose_action(self, states):
        states = torch.tensor(states, dtype=torch.float32, device=device)
        dist = Categorical(self.model.policy(states))
        actions = dist.sample().cpu().detach()
        return actions.numpy()

    def step(self, states, actions, rewards, next_states, dones, train=True):
        self.states.append(states)
        self.actions.append(actions)
        if self.network is not None:
            intrinsic_rewards = self.compute_intrinsic_rewards(next_states)
            writer.add_scalar('intrinsic rewards', np.mean(intrinsic_rewards), self.num//self.n_envs)
            self.intrinsic_rewards.append(intrinsic_rewards)
        self.rewards.append(rewards)
        self.next_states.append(next_states)
        self.dones.append(dones)
        self.counter += 1
        if self.counter % self.horizon == 0 and train:
            self.counter = 0
            self.train()
            self.states.clear()
            self.actions.clear()
            self.rewards.clear()
            self.next_states.clear()
            self.dones.clear()

    def train(self):
        discounted_reward = 0
        rewards = []

        for reward, dones in zip(reversed(self.rewards), reversed(self.dones)):
            discounted_reward = (reward + (self.gamma * discounted_reward)) * (1 - dones)
            rewards.insert(0, discounted_reward)

        self.old_model.load_state_dict(self.model.state_dict())
        states = torch.tensor(np.concatenate(self.states)).to(device=device, dtype=torch.float32).detach()
        actions = torch.tensor(np.concatenate(self.actions)).to(device=device, dtype=torch.long).detach()
        next_states = torch.tensor(np.concatenate(self.next_states)).to(device=device, dtype=torch.float32).detach()
        rewards = torch.tensor(np.concatenate(rewards)).to(device=device, dtype=torch.float32).detach()
        dones = torch.tensor(np.concatenate(self.dones)).to(device=device, dtype=torch.float32).detach()
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        if self.network is not None:
            intrinsic = []
            for reward, dones in zip(reversed(self.intrinsic_rewards), reversed(self.dones)):
                discounted_reward = (reward + (self.i_gamma * discounted_reward)) * (1 - dones)
                intrinsic.insert(0, discounted_reward)

            intrinsic = torch.tensor(np.concatenate(intrinsic)).to(device=device, dtype=torch.float32).detach()

            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            intrinsic = (intrinsic - intrinsic.mean()) / (intrinsic.std())

            # value_target = torch.tensor(rewards).to(device=device, dtype=torch.float32).detach().flatten()

            for _ in range(self.epochs):
                self.update_count += 1

                ind = torch.tensor(np.random.randint(0, self.n_envs * self.horizon, size=self.batch_size)).to(device=device,
                                                                                                              dtype=torch.long)

                advantage = 2*(rewards - self.model.value(states)[0].detach().squeeze(1)) + (intrinsic - self.model.value(states)[1].detach().squeeze(1))

                old_dist = Categorical(self.old_model.policy(states))
                old_log_probs = old_dist.log_prob(actions)
                new_dist = Categorical(self.model.policy(states))
                new_log_probs = new_dist.log_prob(actions)
                entropy = new_dist.entropy()

                ratio = torch.exp(new_log_probs - old_log_probs.detach())
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantage
                loss1 = -torch.min(surr1, surr2)
                loss2 = F.mse_loss(rewards, self.model.value(states)[0].squeeze(1)) + F.mse_loss(intrinsic, self.model.value(states)[1].detach().squeeze(1))
                writer.add_scalar('surrogate loss', -loss1.detach().mean().item(), self.update_count)
                writer.add_scalar('val loss', loss2.detach().mean().item(), self.update_count)
                writer.add_scalar('entropy loss', entropy.detach().mean().item(), self.update_count)
                loss = torch.mean(loss1 - self.entropy * entropy) + self.VF * loss2
                writer.add_scalar('overall_loss', loss.detach().item(), self.update_count)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                predicted = self.predictor(next_states)
                actual = self.network(next_states)
                error = F.mse_loss(predicted, actual)
                self.predictor_optim.zero_grad()
                error.backward()
                self.predictor_optim.step()
        else:
            for _ in range(self.epochs):
                self.update_count += 1

                ind = torch.tensor(np.random.randint(0, self.n_envs * self.horizon, size=self.batch_size)).to(
                    device=device,
                    dtype=torch.long)

                advantage = rewards - self.model.value(states).detach().squeeze(1)

                old_dist = Categorical(self.old_model.policy(states))
                old_log_probs = old_dist.log_prob(actions)
                new_dist = Categorical(self.model.policy(states))
                new_log_probs = new_dist.log_prob(actions)
                entropy = new_dist.entropy()[ind]

                ratio = torch.exp(new_log_probs - old_log_probs.detach())
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantage
                loss1 = -torch.min(surr1, surr2)[ind]
                loss2 = F.mse_loss(rewards[ind], self.model.value(states).squeeze(1)[ind])
                writer.add_scalar('surrogate loss', -loss1.detach().mean().item(), self.update_count)
                writer.add_scalar('val loss', loss2.detach().mean().item(), self.update_count)
                writer.add_scalar('entropy loss', entropy.detach().mean().item(), self.update_count)
                loss = torch.mean(loss1 - self.entropy * entropy) + self.VF * loss2
                writer.add_scalar('overall_loss', loss.detach().item(), self.update_count)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

    def compute_intrinsic_rewards(self, next_state):
        next_state = torch.tensor(next_state).to(device=device, dtype=torch.float32)
        predicted = self.predictor(next_state)
        actual = self.network(next_state)
        error = (predicted-actual).norm(dim=1)
        error = error.detach().cpu().numpy()
        self.num += self.n_envs
        self.sumx += np.sum(error)
        self.sumx2 += np.sum(error**2)
        mean = self.sumx / self.num
        stdev = np.sqrt((self.sumx2 / self.num) - (mean * mean))
        if self.num < 200 * self.n_envs:
            return 0
        else:
            return error / stdev

