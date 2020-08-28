import gym
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
import numpy as np
from DDPG import DDPG
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
env = gym.make('CarRacing-v0')


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.pool1 = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.pool2 = nn.MaxPool2d(5)
        self.lin1 = nn.Linear(800, 128)
        self.lin2 = nn.Linear(128, 28)
        self.lin3 = nn.Linear(31, 1)

    def forward(self, input, action):
        # print("critic input", input.shape)
        # print("critic action", action.shape)
        x = F.relu(self.conv1(input))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = torch.cat((x, action), 1)
        x = self.lin3(x)
        return x


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.pool1 = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.pool2 = nn.MaxPool2d(5)
        self.lin1 = nn.Linear(800, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 3)

    def forward(self, input):
        # print("actor input", input.shape)
        x = F.relu(self.conv1(input))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = torch.tanh(self.lin3(x))
        return x


min = env.action_space.low
max = env.action_space.high
epochs = 5000
agent = DDPG(Actor, Critic, env.action_space)

avg_rewards = []
rewards = []
t = 0
for i in range(epochs):
    state = env.reset()
    done = False
    episodes = []
    episode_reward = 0
    state = state.reshape((3, 96, 96))
    # agent.decrement_std()

    while not done:
        if i % 10 == 0:
            env.render()
        action = agent.choose_action(state.reshape(1, 3, 96, 96), t)
        action = np.clip(action, min, max)
        # action[2] = 0
        # print(action)
        next_state, reward, done, _ = env.step(action)
        next_state = next_state.reshape((3, 96, 96))
        episode_reward += reward
        agent.store(state, action, reward, next_state, done)
        state = next_state
        t += 1
        if t % 8 == 0:
            agent.train()

    rewards.append(episode_reward)
    if i % 10 == 0 and i > 0:
        print('Episode {}: {}, {}'.format(i, np.mean(rewards), agent.noise.sigma))
        avg_rewards.append(np.mean(rewards))
        if np.mean(rewards) >= 900:
            break
        rewards.clear()

torch.save(agent.actor.state_dict(), 'racingActor.pt')
torch.save(agent.critic.state_dict(), 'racingCritic.pt')

plt.plot(range(len(avg_rewards)), avg_rewards)
plt.show()
