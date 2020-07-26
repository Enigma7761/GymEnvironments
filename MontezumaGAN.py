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
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.cuda.empty_cache()

writer = SummaryWriter()
seq_len = 4
num_envs = 16
seed = 3
n = 4
torch.manual_seed(seed)
np.random.seed(seed)
env = make_atari_env('BreakoutNoFrameskip-v4', num_env=num_envs, seed=seed)
env = VecFrameStack(env, n_stack=seq_len)
action_space = 4
max_timesteps = 200000000

filename = "BreakoutExploration"


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(seq_len, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.lin1 = nn.Linear(3136, 512)
        self.q = nn.Linear(512, 1)
        self.lin3 = nn.Linear(512, action_space)

    def forward(self, input):
        x = input.view(-1, seq_len, 84, 84)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 3136)
        x = F.relu(self.lin1(x))
        q = self.q(x)
        action_vals = self.lin3(x)
        return q + (action_vals - torch.mean(action_vals, dim=0))


def produce_onehot(actions):
    actions = actions.view(-1, 1)
    one_hot = torch.zeros((actions.shape[0], action_space)).to(device)
    one_hot.scatter_(1, actions, 1)
    return one_hot


class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.engine = nn.Sequential(
            nn.Conv2d(seq_len, 32, 8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

        self.lstm1 = nn.LSTM(1152+action_space, 256)
        self.hidden_eval = (torch.zeros((1, num_envs, 256)).to(device), torch.zeros((1, num_envs, 256)).to(device))
        self.hidden_train = (torch.zeros((1, num_envs, 256)).to(device), torch.zeros((1, num_envs, 256)).to(device))
        self.batch_norm = nn.BatchNorm1d(256)

        self.renderer = nn.Sequential(
            nn.ConvTranspose2d(256, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(1024, 512, 5, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. (ngf*4) x 9 x 9
            nn.ConvTranspose2d(512, 256, 5, 2, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. (ngf*2) x 21 x 21
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. (ngf) x 42 x 42
            nn.ConvTranspose2d(128, 4, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input, action):
        x = input.view(-1, 4, 84, 84)
        x = self.engine(x)
        x = x.view(-1, 1152)
        action = produce_onehot(action)
        x = torch.cat([x, action.view(-1, action_space)], dim=1)
        if self.training:
            x = x.view(-1, num_envs, 1152+action_space)
            x, self.hidden_train = self.lstm1(x, self.hidden_train)
        else:
            x = x.view(-1, num_envs, 1152+action_space)
            x, self.hidden_eval = self.lstm1(x, self.hidden_eval)
        try:
            x = torch.relu(x)
        except RuntimeError:
            pass
        x = x.view(-1, 256, 1, 1)
        x = self.renderer(x)
        return x.view(-1, 4, 84, 84)

    def choose_action(self, input, average_state):
        x = input.view(-1, 4, 84, 84)
        actions = torch.tensor(list(range(0, action_space))*num_envs, dtype=torch.long, device=device).view(-1, 1)
        actions = produce_onehot(actions)
        x = self.engine(x)
        x = x.view(num_envs, 1152)
        x = torch.repeat_interleave(x, action_space, 0)
        x = torch.cat([x, actions], dim=1)
        x = x.view(1, -1, 1152+action_space)
        hidden_temp = (torch.repeat_interleave(self.hidden_eval[0], action_space, 1), torch.repeat_interleave(self.hidden_eval[1], action_space, 1))
        x, _ = self.lstm1(x, hidden_temp)
        x = F.relu(x)
        x = x.view(-1, 256, 1, 1)
        x = self.renderer(x)
        x = x.view(-1, 4, 84, 84)
        average_state = torch.repeat_interleave(average_state, action_space*num_envs, 0)
        differences = torch.mean((x - average_state) ** 2, dim=(1, 2, 3))
        differences = differences.view(action_space, -1)
        actions = torch.argmax(differences, dim=0)
        return actions.detach().cpu().numpy()

    def reset(self):
        if not self.training:
            self.hidden_eval = (torch.zeros((1, num_envs, 256)).to(device), torch.zeros((1, num_envs, 256)).to(device))
        else:
            self.hidden_train = (torch.zeros((1, num_envs, 256)).to(device), torch.zeros((1, num_envs, 256)).to(device))


predictor = Predictor().to(device)
predictor_optim = torch.optim.Adam(predictor.parameters(), lr=2e-4, betas=(0.5, 0.999))

epochs = 5000
agent = DQN(Model, gamma=0.99, learning_rate=0.0000625)

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
time_steps = []
rewards_deque = deque(maxlen=100)
n_rewards = deque(maxlen=n)
rewards = []
episode_loss = []
t = 1
epsilon = 1

criterion = nn.BCELoss()
states = deque(maxlen=10)
next_states = deque(maxlen=10)
actions = deque(maxlen=10)
hidden = predictor.hidden_eval
average_states = deque(maxlen=100)

state = np.swapaxes(env.reset(), 1, 3).astype(np.uint8)

i = 0
while t * num_envs < max_timesteps:
    i += 1
    episode_reward = 0
    done = [False for _ in range(num_envs)]

    losses = []

    action = 0
    if t > 10000000:
        epsilon = 0.05
    else:
        epsilon = max(0.1, epsilon * 0.997)
    for _ in range(n):
        n_rewards.append(np.zeros(num_envs))
    for _ in range(num_envs):
        while True:
            average_states.append(state)
            # env.render(mode="human")
            states.append(state)
            if np.random.random() > epsilon:
                action = agent.choose_action(state)
            else:
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    hidden = predictor.hidden_eval
                    predictor.eval()
                    state_tensor = torch.from_numpy(state).to(device, dtype=torch.float32)
                    average_state = np.array(average_states).mean((0, 1))
                    average_state = average_state.reshape(1, 4, 84, 84)
                    average_state = torch.from_numpy(average_state).to(device, dtype=torch.float32)
                    action = predictor.choose_action(state_tensor, average_state)
                    predictor.hidden_eval = hidden
                    del state_tensor
                    del average_state
            with torch.no_grad():
                torch.cuda.empty_cache()
                action_tensor = torch.from_numpy(action).to(device, dtype=torch.long)
                state_tensor = torch.from_numpy(state).to(device, dtype=torch.float32)
                predictor.train(False)
                temp = predictor(state_tensor, action_tensor).view(-1, 4, 84, 84).detach().cpu().numpy()
                predictor.train()
                del action_tensor
                del state_tensor
            # cv2.imshow('window', (temp[0, -1, :, :].reshape(84, 84, 1) * 128).astype(np.uint8))
            # cv2.imshow('state', (state[0, -1, :, :].reshape(84, 84, 1) * 128).astype(np.uint8))
            if cv2.waitKey(25) == ord('q'):
                break
            next_state, reward, done, _ = env.step(action)
            n_rewards.append(reward)
            temp = np.array(n_rewards)
            next_state = np.swapaxes(next_state, 1, 3).astype(np.uint8)
            episode_reward += np.mean(reward)
            for j in range(num_envs):
                agent.store(state[j, :, :, :], action[j], temp[:, j], next_state[j, :, :, :], done[j])
            actions.append(action)
            next_states.append(next_state)
            if True in done:
                done = torch.from_numpy(done).to(device, dtype=torch.float32).view(1, -1, 1)
                predictor.hidden_eval = (predictor.hidden_eval[0] * done, predictor.hidden_eval[1] * done)
                break
            if len(states) == states.maxlen:
                predictor.train()
                torch.cuda.empty_cache()
                states_tensor = torch.as_tensor(states).to(device, dtype=torch.float32).view(-1, 4, 84, 84)
                next_states_tensor = torch.as_tensor(next_states).to(device, dtype=torch.float32).view(-1, 4, 84, 84)
                actions_tensor = torch.as_tensor(actions).to(device, dtype=torch.long).view(-1, 1)
                for _ in range(max(4, 4)):
                    torch.cuda.empty_cache()
                    predictor.hidden_train = hidden
                    predictor_optim.zero_grad()
                    predictor.train()
                    fake = predictor(states_tensor, actions_tensor).view(-1, 4, 84, 84)
                    label = torch.full((states_tensor.shape[0],), 1, device=device)

                    predictor_loss = F.mse_loss(fake, next_states_tensor)
                    try:
                        predictor_loss.backward()
                    except RuntimeError:
                        predictor.train()
                    predictor_optim.step()
                    del fake
                    del label
                    del predictor_loss
                states.clear()
                next_states.clear()
                actions.clear()
                hidden = predictor.hidden_eval
                # del output
                del states_tensor
                del next_states_tensor
                del actions_tensor
            state = next_state
            if t * num_envs > 50000:
                torch.cuda.empty_cache()
                for _ in range(4):
                    loss = agent.train()
                    if loss is not None:
                        losses.append(loss)
                    if t % (32000 // num_envs) == 0:
                        agent.copy_weights()
                    # agent.soft_update()
            t += 1

    if len(losses) > 0:
        episode_loss.append(np.mean(losses))
        writer.add_scalar('Loss', np.mean(losses), i)
    rewards.append(episode_reward)
    rewards_deque.append(episode_reward)
    if i % 20 == 0:
        cv2.imwrite("AverageMontezuma.png", np.array(average_states).mean((0, 1, 2)).reshape(84, 84, 1))
    print('\rEpisode {}, Time {}: {:.3f},  Avg:{:.3f}, Noise {:.3f}, Epsilon {:.2f}'.format(i, t * num_envs, episode_reward,
                                                                                            np.mean(rewards_deque),
                                                                                            agent.param_noise,
                                                                                            epsilon), end='')
    if i % 10 == 0:
        avg_rewards.append(np.mean(rewards_deque))
        writer.add_scalar('Scaled Rewards', np.mean(rewards_deque), t*num_envs)
        time_steps.append(t * num_envs)
    if i % 100 == 0:
        print(
            '\rEpisode {}, Time {}: {:.3f},  Avg:{:.3f}, Noise {:.3f}, Epsilon {:.2f}'.format(i, t * num_envs, episode_reward,
                                                                                              np.mean(
                                                                                                  rewards_deque),
                                                                                              agent.param_noise,
                                                                                              epsilon))
        torch.save(agent.value.state_dict(), 'StateDicts/' + filename + '.pt')
        torch.save(predictor.state_dict(), 'Generator.pt')
        rewards.clear()

fig, axs = plt.subplots(2, 1)
axs[0].plot(time_steps, avg_rewards)
axs[1].plot(range(len(episode_loss)), episode_loss)
plt.show()
