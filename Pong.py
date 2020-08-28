import gym
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
import numpy as np
from DQN import DQN
import torch
from collections import deque
import cv2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.cuda.empty_cache()

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
env = gym.make('Pong-v0')
filename = "Pong"

seq_len = 4


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(seq_len, 16, 3)
        self.pool1 = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(5)
        self.lin1 = nn.Linear(768, 256)
        self.lnorm1 = nn.LayerNorm(256)
        self.lin2 = nn.Linear(256, 100)
        self.lnorm2 = nn.LayerNorm(100)
        self.q = nn.Linear(100, 1)
        self.lin3 = nn.Linear(100, 4)

    def forward(self, input):
        x = input.view(-1, seq_len, 105, 80)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 768)
        x = F.relu(self.lnorm1(self.lin1(x)))
        x = F.relu(self.lnorm2(self.lin2(x)))
        q = self.q(x)
        action_vals = self.lin3(x)
        return q + action_vals - torch.mean(action_vals, dim=0)


epochs = 5000
agent = DQN(Model)

try:
    agent.value.load_state_dict(torch.load('StateDicts/' + filename + '.pt'))
    agent.target1.load_state_dict(torch.load('StateDicts/' + filename + '.pt'))
    agent.target2.load_state_dict(torch.load('StateDicts/' + filename + '.pt'))
    print("Files loaded")
except FileNotFoundError:
    print("Files not found")
    pass
except RuntimeError:
    print("New Model")
    pass

avg_rewards = []
rewards_deque = deque(maxlen=100)
state_deque = deque(maxlen=seq_len)
next_state_deque = deque(maxlen=seq_len)
rewards = []
t = 0
# epsilon = 1

try:
    for i in range(1, epochs+1):
        state = env.reset()
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = cv2.resize(state, (80, 105))

        states = []
        done = False
        episode_reward = 0
        # epsilon = max(0.05, epsilon * 0.99)
        action = 0
        for _ in range(seq_len):
            state_deque.append(np.zeros((105, 80)))
            next_state_deque.append(np.zeros((105, 80)))
        while not done:
            if i % 100 == 0:
                env.render()

            state_deque.append(state)

            temp = np.vstack(state_deque)

            if t % seq_len == 0:
                states.append(temp)
                # if np.random.random() > epsilon:
                action = agent.choose_action(temp)
                # else:
                #     action = np.random.randint(0, 4)
            next_state, reward, done, _ = env.step(action)

            next_state = cv2.cvtColor(next_state, cv2.COLOR_RGB2GRAY)

            next_state = cv2.resize(next_state, (80, 105))

            next_state_deque.append(next_state)
            episode_reward += reward
            if t % seq_len == 0:
                agent.store(temp, action, reward,  np.vstack(next_state_deque), done)
            state = next_state
            if t % 1 == 0:
                agent.train()
                agent.soft_update()

            t += 1
        torch.cuda.empty_cache()

        states = torch.tensor(states, dtype=torch.float32, device=device)
        prev_actions = agent.value(states)
        agent.apply_param_noise()
        new_actions = agent.value(states)
        distance = torch.sqrt(F.mse_loss(prev_actions, new_actions)).detach().cpu().item()
        agent.adapt_noise(distance)
        rewards.append(episode_reward)
        rewards_deque.append(episode_reward)
        print('\rEpisode {}: {:.3f},  Avg:{:.3f}, Noise {:.3f}'.format(i, episode_reward, np.mean(rewards_deque),
                                                                       agent.param_noise), end='')
        if i % 10 == 0:
            avg_rewards.append(np.mean(rewards))
        if i % 100 == 0:
            print('\rEpisode {}: {:.3f},  Avg:{:.3f}, Noise {:.3f}'.format(i, episode_reward,
                                                                           np.mean(rewards_deque), agent.param_noise))
            torch.save(agent.value.state_dict(), 'StateDicts/' + filename + '.pt')
            rewards.clear()
except KeyboardInterrupt:
    torch.save(agent.value.state_dict(), 'StateDicts/' + filename + '.pt')


plt.plot(range(len(avg_rewards)), avg_rewards)
plt.show()
