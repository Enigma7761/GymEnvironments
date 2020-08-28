import gym
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
import numpy as np
from DQN import DQN
from R2D2 import R2D2
import torch
from collections import deque
import time
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.cuda.empty_cache()

seq_len = 4
batch_size = 64
num_envs = 32
seed = 0
n = 8
torch.manual_seed(seed)
np.random.seed(seed)
env = make_atari_env('BreakoutNoFrameskip-v4', num_env=num_envs, seed=seed)
env = VecFrameStack(env, n_stack=seq_len)

filename = "Breakout2"


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(seq_len, 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.lstm1 = nn.Linear(2592, 512)
        self.lin1 = nn.Linear(512, 128)
        self.q = nn.Linear(128, 1)
        self.lin2 = nn.Linear(128, 4)
        self.training_hidden = (
        torch.zeros((num_envs, seq_len, 128)).to(device), torch.zeros((num_envs, seq_len, 128)).to(device))
        self.eval_hidden = (
        torch.zeros((batch_size, seq_len, 128)).to(device), torch.zeros((batch_size, seq_len, 128)).to(device))

    def forward(self, input, seq_len=1):
        x = input.view(-1, seq_len, 84, 84)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, seq_len, 2592)
        if self.training:
            x, self.training_hidden = F.relu(self.lstm1(x, self.training_hidden))
        else:
            x, self.eval_hidden = F.relu(self.lstm1(x, self.eval_hidden))
        x = F.relu(self.lin1(x))
        x = x.view(-1, seq_len, 2592)
        q = self.q(x)
        action_vals = self.lin2(x)
        return q + (action_vals - torch.mean(action_vals, dim=0))

    def reset_hidden(self, index=None):
        if self.training:
            if index is not None:
                self.training_hidden[index, :, :] = 0
            else:
                self.training_hidden[:] = 0
        else:
            if index is not None:
                self.eval_hidden[index, :, :] = 0
            else:
                self.eval_hidden[:] = 0


epochs = 3500
agent = R2D2(Model, minibatch_size=batch_size)

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
sequence = []

state = np.swapaxes(env.reset(), 1, 3) / 128

try:
    for i in range(1, epochs + 1):
        # try:
        # except RuntimeError:
        #     env.close()
        #     env = make_atari_env('BreakoutNoFrameskip-v4', num_env=num_envs, seed=np.random.randint(0, 100))
        #     env = VecFrameStack(env, n_stack=seq_len)
        #     state = env.reset().reshape(num_envs, seq_len, 84, 84)/128

        episode_reward = 0
        done = np.array([False for _ in range(num_envs)])

        losses = []

        action = 0
        epsilon = max(0.05, epsilon * 0.993)
        for _ in range(n):
            n_rewards.append(np.zeros(num_envs))
        for _ in range(num_envs):
            while True:
                # cv2.imshow('window', state[0, 0, :, :].reshape(84, 84)*128)
                # if cv2.waitKey(25) == ord('q'):
                #     break
                #
                # env.render(mode="human")
                if np.random.random() > epsilon:
                    action = agent.choose_action(state)
                else:
                    action = np.random.randint(2, 4, size=num_envs)
                next_state, reward, done, _ = env.step(action)
                sequence.append([state, action, reward])

                n_rewards.append(reward)
                temp = np.array(n_rewards)
                if True in done:
                    indices = np.where(done is True)[0]
                    agent.value.eval()
                    agent.value.reset_hidden(indices)
                    agent.value.train()
                    break
                next_state = np.swapaxes(next_state, 1, 3) / 128
                episode_reward += np.mean(reward)
                for j in range(num_envs):
                    agent.store(state[j, :, :, :], action[j], temp[:, j], next_state[j, :, :, :], done[j])
                state = next_state
                if t * num_envs > 20000:
                    torch.cuda.empty_cache()
                    for _ in range(1):
                        loss = agent.train()
                        if loss is not None:
                            losses.append(loss)
                        agent.soft_update()
                t += 1

                if True in done:
                    break
                torch.cuda.empty_cache()
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
