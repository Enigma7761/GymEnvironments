import gym
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
import numpy as np
from DQN import DQN
import torch
from collections import deque
import keyboard

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.cuda.empty_cache()

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
env = gym.make('CartPole-v0')
filename = "CartPole"

action_space = 2
state_space = 4

seq_len = 2


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lin1 = nn.Linear(state_space * seq_len, 64)
        self.lnorm1 = nn.LayerNorm(64)
        self.lin2 = nn.Linear(64, 32)
        self.lnorm2 = nn.LayerNorm(32)
        self.q = nn.Linear(32, 1)
        self.lin3 = nn.Linear(32, action_space)

    def forward(self, input):
        x = input.view(-1, state_space * seq_len)
        x = F.relu(self.lnorm1(self.lin1(x)))
        x = F.relu(self.lnorm2(self.lin2(x)))
        q = self.q(x)
        action_vals = self.lin3(x)
        return q + action_vals - torch.mean(action_vals)


epochs = 5000
agent = DQN(Model, minibatch_size=64, param_noise=0.1)

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
epsilon = 1

try:
    for i in range(1, epochs+1):
        state = env.reset()
        states = []
        done = False
        episode_reward = 0
        action = 0
        epsilon = max(0.05, epsilon*0.95)
        for _ in range(seq_len):
            state_deque.append(np.zeros(state_space))
            next_state_deque.append(np.zeros(state_space))
        while not done:
            if i % 100 == 0:
                env.render()

            state_deque.append(state.reshape(state_space))
            if t % seq_len == 0:
                states.append(np.array(state_deque).reshape(seq_len * state_space))
                if np.random.random() > epsilon:
                    action = agent.choose_action(np.array(state_deque).reshape(1, seq_len * state_space))
                else:
                    action = np.random.randint(0, 2)
            next_state, reward, done, _ = env.step(action)
            next_state_deque.append(next_state.reshape(state_space))
            episode_reward += reward
            agent.store(state=np.array(state_deque).reshape(state_space * seq_len), action=action, reward=reward,
                        next_state=np.array(next_state_deque).reshape(state_space * seq_len), done=done)
            state = next_state
            if t % 1 == 0:
                agent.train()
                agent.soft_update()
            t += 1
        torch.cuda.empty_cache()

        # states = torch.tensor(states, dtype=torch.float32, device=device)
        # prev_actions = agent.value(states)
        # agent.apply_param_noise()
        # new_actions = agent.value(states)
        # distance = torch.sqrt(F.mse_loss(prev_actions, new_actions)).detach().cpu().item()
        # agent.adapt_noise(distance)
        rewards.append(episode_reward)
        rewards_deque.append(episode_reward)
        print('\rEpisode {}: {:.3f},  Avg:{:.3f}, Noise {:.3f}'.format(i, episode_reward, np.mean(rewards_deque),
                                                                       epsilon), end='')
        if i % 10 == 0:
            avg_rewards.append(np.mean(rewards))
        if i % 100 == 0:
            print('\rEpisode {}: {:.3f},  Avg:{:.3f}, Noise {:.3f}'.format(i, episode_reward,
                                                                           np.mean(rewards_deque), epsilon))
            torch.save(agent.value.state_dict(), 'StateDicts/' + filename + '.pt')
            rewards.clear()
except KeyboardInterrupt:
    torch.save(agent.value.state_dict(), 'StateDicts/' + filename + '.pt')


plt.plot(range(len(avg_rewards)), avg_rewards)
plt.show()
