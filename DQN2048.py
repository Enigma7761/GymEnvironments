import gym
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
import numpy as np
from DQN import DQN
import torch
from collections import deque
import time
import gym2048
from stable_baselines3.common.cmd_util import make_vec_env
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.cuda.empty_cache()

writer = SummaryWriter()
seq_len = 1
batch_size = 128
num_envs = 16
seed = 0
n = 1
torch.manual_seed(seed)
np.random.seed(seed)
env = make_vec_env('gym2048-v0', n_envs=num_envs)

filename = "2048"


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lin1 = nn.Linear(16*18, 1024)
        self.lin2 = nn.Linear(1024, 512)
        self.lin3 = nn.Linear(512, 512)
        self.lin4 = nn.Linear(512, 512)
        self.lin5 = nn.Linear(512, 256)
        self.q = nn.Linear(256, 1)
        self.lin6 = nn.Linear(256, 4)

    def forward(self, input, seq_len=1):
        x = input.reshape(-1, 16*18)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.relu(self.lin4(x))
        x = F.relu(self.lin5(x))
        q = self.q(x)
        action_vals = self.lin6(x)
        return q + (action_vals - torch.mean(action_vals, dim=0))


epochs = 10000
agent = DQN(Model, minibatch_size=batch_size, gamma=0.99, learning_rate=0.0000625, tau=1e-5)

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

state = np.swapaxes(env.reset(), 1, 3)
state = state.reshape(num_envs, -1)
state[state == 0] = 1
state = np.log2(state)
state = state.astype(int)
state = np.eye(18)[state].reshape(num_envs, -1)

try:
    for i in range(1, epochs + 1):
        # try:
        # except RuntimeError:
        #     env.close()
        #     env = make_atari_env('BreakoutNoFrameskip-v4', num_env=num_envs, seed=np.random.randint(0, 100))
        #     env = VecFrameStack(env, n_stack=seq_len)
        #     state = env.reset().reshape(num_envs, seq_len, 84, 84)/128
        episode_reward = 0
        scores = 0
        done = np.array([False for _ in range(num_envs)])

        losses = []

        action = 0
        epsilon = max(0.05, epsilon * 0.999)
        for _ in range(n):
            n_rewards.append(np.zeros(num_envs))
        for _ in range(1):
            while True:
                # cv2.imshow('window', state[0, 0, :, :].reshape(84, 84)*128)
                # if cv2.waitKey(25) == ord('q'):
                #     break
                #
                # env.render()
                if np.random.random() > epsilon:
                    action = agent.choose_action(state)
                else:
                    action = np.random.randint(0, 4, size=num_envs)

                next_state, reward, done, info = env.step(action)

                n_rewards.append(reward)
                temp = np.array(n_rewards)
                if True in done:
                    state = np.swapaxes(env.reset(), 1, 3)
                    state = state.reshape(num_envs, -1)
                    state[state == 0] = 1
                    state = np.log2(state)
                    state = state.astype(int)
                    state = np.eye(18)[state].reshape(num_envs, -1)
                    tiles = []
                    info_done = np.asarray(info)[done].tolist()
                    for l in range(len(info_done)):
                        scores += info_done[l]["score"]
                        tiles.append(info_done[l]["highest_tile"])
                        # print(info_done[l]["terminal_observation"].squeeze())
                    scores /= len(info_done)
                    writer.add_scalar('Highest Tile', np.max(tiles), t * num_envs)
                    break
                next_state = np.swapaxes(next_state, 1, 3)
                next_state = next_state.reshape(num_envs, -1)
                next_state[next_state == 0] = 1
                next_state = np.log2(next_state)
                next_state = next_state.astype(int)
                next_state = np.eye(18)[next_state].reshape(num_envs, -1)
                episode_reward += np.mean(reward)
                for j in range(num_envs):
                    agent.store(state[j, :], np.squeeze(action[j]).item(), temp[:, j], next_state[j, :], np.asarray(done[j]))
                state = next_state
                if t * num_envs > 150000:
                    torch.cuda.empty_cache()
                    for _ in range(6):
                        loss = agent.train()
                        if loss is not None:
                            losses.append(loss)
                        agent.soft_update()
                    if t % (64000 // num_envs) == 0:
                        agent.copy_weights()
                t += 1

                torch.cuda.empty_cache()
        episode_loss.append(np.mean(losses))
        rewards.append(scores)
        rewards_deque.append(scores)
        if len(losses) > 0:
            episode_loss.append(np.mean(losses))
            writer.add_scalar('Loss', np.mean(losses), t)
        print('\rEpisode {}, Time {}: {:.3f},  Avg:{:.3f}, Noise {:.3f}, Epsilon {:.2f}'.format(i, t, scores,
                                                                                                np.mean(rewards_deque),
                                                                                                agent.param_noise,
                                                                                                epsilon), end='')
        if i % 10 == 0:
            avg_rewards.append(np.mean(rewards))
            writer.add_scalar('Average Score', np.mean(rewards_deque), t * num_envs)
        if i % 100 == 0:
            print(
                '\rEpisode {}, Time {}: {:.3f},  Avg:{:.3f}, Noise {:.3f}, Epsilon {:.2f}'.format(i, t, scores,
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
