import gym
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
import numpy as np
from TD3 import TD3
import torch
from collections import deque

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
env = gym.make('Pendulum-v0')
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
# env = gym.make('BipedalWalker-v2')

action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.lin1 = nn.Linear(action_dim + state_dim, state_dim*32)
        self.lnorm1 = nn.LayerNorm(state_dim*32)
        self.lin2 = nn.Linear(state_dim*32, action_dim*48)
        self.lnorm2 = nn.LayerNorm(action_dim*48)
        self.lin3 = nn.Linear(action_dim*48, 1)

    def forward(self, input, action):
        action = action.reshape((action.shape[0], action_dim))
        x = torch.cat([input, action], 1)
        x = F.relu(self.lnorm1(self.lin1(x)))
        x = F.relu(self.lnorm2(self.lin2(x)))
        x = self.lin3(x)
        return x


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.lin1 = nn.Linear(state_dim, state_dim*32)
        self.lnorm1 = nn.LayerNorm(state_dim*32)
        self.lin2 = nn.Linear(state_dim*32, action_dim*48)
        self.lnorm2 = nn.LayerNorm(action_dim*48)
        self.lin3 = nn.Linear(action_dim*48, action_dim)

    def forward(self, input):
        x = F.relu(self.lnorm1(self.lin1(input)))
        x = F.relu(self.lnorm2(self.lin2(x)))
        x = torch.tanh(self.lin3(x))
        return x


min = env.action_space.low
max = env.action_space.high
epochs = 5000
agent = TD3(Actor, Critic, env.action_space, seed=seed)

avg_rewards = []
rewards_deque = deque(maxlen=100)
rewards = []
t = 0

for i in range(1, epochs+1):
    state = env.reset()
    states = []
    done = False
    episodes = []
    episode_reward = 0
    while not done:
        states.append(state)
        if i % 100 == 0:
            env.render()
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        agent.store(state, action, reward, next_state, done)
        state = next_state
        t += 1
        agent.train(t)

    states = torch.tensor(states, dtype=torch.float32, device=device)
    prev_actions = agent.actor(states)
    agent.apply_param_noise()
    new_actions = agent.actor(states)
    distance = torch.sqrt(F.mse_loss(prev_actions, new_actions)).detach().cpu().item()
    agent.adapt_noise(distance)
    rewards.append(episode_reward)
    rewards_deque.append(episode_reward)
    print('\rEpisode {}: {},  Avg:{}'.format(i, episode_reward, np.mean(rewards_deque)), end='')
    if i % 100 == 0 and i > 0:
        print('\rEpisode {}: {},  Avg:{}'.format(i, episode_reward, np.mean(rewards)))

        avg_rewards.append(np.mean(rewards))
        if np.mean(rewards) >= 300:
            break
    #     torch.save(agent.actor.state_dict(), 'StateDicts/pendulumActor.pt')
    #     torch.save(agent.critic.state_dict(), 'StateDicts/pendulumCritic.pt')

        rewards.clear()

plt.plot(range(len(avg_rewards)), avg_rewards)
plt.show()
