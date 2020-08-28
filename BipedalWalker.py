import gym
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
import numpy as np
from TD3 import TD3
import torch
from collections import deque
import keyboard

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
seed = 0
times = 2
torch.manual_seed(seed)
np.random.seed(seed)
env = gym.make('BipedalWalkerHardcore-v2') #
filename = "walkerHardcore"

action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.lin1 = nn.Linear(action_dim + state_dim * times, state_dim * 32)
        self.lnorm1 = nn.LayerNorm(state_dim * 32)
        self.lin2 = nn.Linear(state_dim * 32, action_dim * 48)
        self.lnorm2 = nn.LayerNorm(action_dim * 48)
        self.lin3 = nn.Linear(action_dim * 48, 1)

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
        self.lin1 = nn.Linear(state_dim * times, state_dim * 32)
        self.lnorm1 = nn.LayerNorm(state_dim * 32)
        self.lin2 = nn.Linear(state_dim * 32, action_dim * 48)
        self.lnorm2 = nn.LayerNorm(action_dim * 48)
        self.lin3 = nn.Linear(action_dim * 48, action_dim)

    def forward(self, input):
        x = F.relu(self.lnorm1(self.lin1(input)))
        x = F.relu(self.lnorm2(self.lin2(x)))
        x = torch.tanh(self.lin3(x))
        return x


min = env.action_space.low
max = env.action_space.high
epochs = 5000
agent = TD3(Actor, Critic, env.action_space, seed=seed)

try:
    agent.actor.load_state_dict(torch.load('StateDicts/' + filename + 'Actor.pt'))
    agent.actor_target.load_state_dict(torch.load('StateDicts/' + filename + 'Actor.pt'))
    agent.critic1.load_state_dict(torch.load('StateDicts/' + filename + 'Critic1.pt'))
    agent.critic2.load_state_dict(torch.load('StateDicts/' + filename + 'Critic2.pt'))
    agent.critic_target1.load_state_dict(torch.load('StateDicts/' + filename + 'Critic1.pt'))
    agent.critic_target2.load_state_dict(torch.load('StateDicts/' + filename + 'Critic2.pt'))
    print("Files loaded")
except FileNotFoundError:
    print("Files not found")
    pass
except RuntimeError:
    print("New Model")

avg_rewards = []
rewards_deque = deque(maxlen=100)
state_deque = deque(maxlen=times)
next_state_deque = deque(maxlen=times)
rewards = []
t = 0
base_trained = False

try:
    for i in range(1, epochs+1):
        if i % 3 != 0 and base_trained:
            env = gym.make('BipedalWalkerHardcore-v2')
        else:
            env = gym.make('BipedalWalker-v2')
        state = env.reset()
        states = []
        done = False
        episode_reward = 0
        for _ in range(times):
            state_deque.append(np.zeros(state.shape))
            next_state_deque.append(np.zeros(state.shape))
        next_state_deque.append(state)
        while not done:
            if i % 100 == 0:
                env.render()

            state_deque.append(state)
            temp = np.array(state_deque)
            temp = temp.reshape(temp.size)
            states.append(temp)

            action = agent.choose_action(temp)
            next_state, reward, done, _ = env.step(action)
            next_state_deque.append(next_state)
            temp_next = np.array(next_state_deque)
            temp_next = temp_next.reshape(temp_next.size)
            episode_reward += reward
            agent.store(temp, action, reward, temp_next, done)
            state = next_state
            t += 1
            agent.train(t)
        env.close()

        states = torch.tensor(states, dtype=torch.float32, device=device)
        prev_actions = agent.actor(states)
        agent.apply_param_noise()
        new_actions = agent.actor(states)
        distance = torch.sqrt(F.mse_loss(prev_actions, new_actions)).detach().cpu().item()
        agent.adapt_noise(distance)
        rewards.append(episode_reward)
        rewards_deque.append(episode_reward)
        print('\rEpisode {}: {:.3f},  Avg:{:.3f}, Noise{:.3f}'.format(i, episode_reward, np.mean(rewards_deque), agent.param_noise), end='')
        if i % 10 == 0:
            avg_rewards.append(np.mean(rewards))
        if i % 100 == 0:
            print('\rEpisode {}: {:.3f},  Avg:{:.3f}, Noise{:.3f}'.format(i, episode_reward, np.mean(rewards_deque), agent.param_noise))
            torch.save(agent.actor.state_dict(), 'StateDicts/' + filename + 'Actor.pt')
            torch.save(agent.critic1.state_dict(), 'StateDicts/' + filename + 'Critic1.pt')
            torch.save(agent.critic2.state_dict(), 'StateDicts/' + filename + 'Critic2.pt')
            if np.mean(rewards) >= 300:
                break
            if np.mean(rewards) >= 200 and not base_trained:
                base_trained = True
            rewards.clear()
except KeyboardInterrupt:
    pass


plt.plot(range(len(avg_rewards)), avg_rewards)
plt.legend()
plt.show()
