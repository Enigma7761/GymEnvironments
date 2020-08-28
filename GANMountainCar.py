import gym
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
import numpy as np
from DQN import DQN
import torch
from collections import deque
import cv2
import time
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.cuda.empty_cache()

seed = 4
n = 8
torch.manual_seed(seed)
np.random.seed(seed)
env = gym.make('MountainCar-v0')

filename = "MountainCar"


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lin1 = nn.Linear(2, 64)
        self.lin2 = nn.Linear(64, 64)
        self.q = nn.Linear(64, 1)
        self.lin3 = nn.Linear(64, 3)

    def forward(self, input):
        x = torch.relu(self.lin1(input))
        x = torch.relu(self.lin2(x))
        q = self.q(x)
        x = self.lin3(x)
        return q - (x - torch.mean(x, dim=0))


class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.lin1 = nn.Linear(5, 64)
        self.lin2 = nn.Linear(64, 64)
        self.lin3 = nn.Linear(64, 2)

    def forward(self, state, action):
        x = torch.cat(state, action, dim=0)
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
        x = self.lin3(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.lin1 = nn.Linear(2, 64)
        self.lin2 = nn.Linear(64, 64)
        self.lin3 = nn.Linear(64, 1)

    def forward(self, state):
        x = torch.relu(self.lin1(state))
        x = torch.relu(self.lin2(x))
        x = torch.sigmoid(self.lin3(x))
        return x


predictor = Predictor().to(device)
predictor_optim = torch.optim.Adam(predictor.parameters(), lr=1e-4)
discriminator = Discriminator().to(device)
discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
epochs = 3000
agent = DQN(Model)

try:
    agent.value.load_state_dict(torch.load('StateDicts/' + filename + '.pt'))
    agent.copy_weights()
    print("Files loaded")
except FileNotFoundError:
    print("Files not found")
    pass
except RuntimeError:
    print("New Model")
    pass

avg_rewards = []
rewards_deque = deque(maxlen=100)
n_rewards = deque(maxlen=n)
rewards = []
episode_loss = []
t = 0
epsilon = 1

try:
    for i in range(1, epochs + 1):
        state = env.reset()

        episode_reward = 0

        losses = []

        action = 0
        epsilon = max(0.05, epsilon * 0.991)
        for _ in range(n):
            n_rewards.append(np.zeros(1))
        while True:
            if np.random.random() > epsilon:
                action = agent.choose_action(state)
            else:
                action = np.random.randint(0, 3)
            next_state, reward, done, _ = env.step(action)
            n_rewards.append(reward)
            temp = np.array(n_rewards)
            if True in done:
                break
            episode_reward += np.mean(reward)
            agent.store(state[0, :, :, :], action[0], temp[:, 0], next_state[0, :, :, :], done[0])
            state = next_state
            torch.cuda.empty_cache()
            loss = agent.train()
            if loss is not None:
                losses.append(loss)
            agent.soft_update()
            t += 1

            if True in done:
                break
            torch.cuda.empty_cache()
            states, actions, rewards, next_states, dones = DQN.replay.sample()
            states = torch.tensor(states, device=device, dtype=torch.float32)
            next_states = torch.tensor(next_states, device=device, dtype=torch.float32)
            actions = torch.tensor(actions, device=device, dtype=torch.long)
            for i in range(2):
                disc_loss = torch.mean(torch.log(discriminator(states)) + torch.log(discriminator(next_states)) + \
                            torch.log(1 - discriminator(predictor(states, actions))), dim=1)
                discriminator_optim.zero_grad()
                disc_loss.backward()
                discriminator_optim.step()
            predictor_loss = torch.mean(torch.log(1 - discriminator(predictor(states, actions))), dim=1) + \
                             F.mse_loss(next_states, predictor(states, actions))
            predictor_optim.zero_grad()
            predictor_loss.backward()
            predictor_optim.step()

        episode_loss.append(np.mean(losses))
        rewards.append(episode_reward)
        rewards_deque.append(episode_reward)
        print('\rEpisode {}, Time {}: {:.3f},  Avg:{:.3f}, Noise {:.3f}, Epsilon {:.2f}'.format(i, t, episode_reward,
                                                                                                np.mean(rewards_deque),
                                                                                                agent.param_noise,
                                                                                                epsilon), end='')
        if i % 10 == 0:
            avg_rewards.append(np.mean(rewards))
        if i % 100 == 0:
            print(
                '\rEpisode {}, Time {}: {:.3f},  Avg:{:.3f}, Noise {:.3f}, Epsilon {:.2f}'.format(i, t, episode_reward,
                                                                                                  np.mean(
                                                                                                      rewards_deque),
                                                                                                  agent.param_noise,
                                                                                                  epsilon))
            torch.save(agent.value.state_dict(), 'StateDicts/' + filename + '.pt')
            rewards.clear()

except KeyboardInterrupt:
    torch.save(agent.value.state_dict(), 'StateDicts/' + filename + '.pt')
torch.save(agent.value.state_dict(), 'StateDicts/' + filename + '.pt')

fig, axs = plt.subplots(2, 1)
axs[0].plot(range(len(avg_rewards)), avg_rewards)
axs[1].plot(range(len(episode_loss)), episode_loss)
plt.show()