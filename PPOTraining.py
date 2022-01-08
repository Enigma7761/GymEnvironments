import gym
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
import numpy as np
from DQN import DQN
from PPO import PPO, writer
import torch
from collections import deque
import time
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.cmd_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.cuda.empty_cache()

seq_len = 4
num_envs = 32
seed = 0
n_stack = 4
torch.manual_seed(seed)
np.random.seed(seed)
env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=num_envs, seed=seed, wrapper_kwargs={'clip_reward': False})
env = VecFrameStack(env, n_stack=n_stack)

filename = "Breakout_PPO1"


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.conv11 = nn.Conv2d(n_stack, 32, 8, stride=4)
        self.conv12 = nn.Conv2d(32, 64, 4, stride=2)
        self.lin11 = nn.Linear(5184, 768)
        self.lin12 = nn.Linear(768, 400)

        self.conv21 = nn.Conv2d(n_stack, 32, 8, stride=4)
        self.conv22 = nn.Conv2d(32, 64, 4, stride=2)
        self.lin21 = nn.Linear(5184, 768)
        self.lin22 = nn.Linear(768, 400)

        self.actor = nn.Linear(400, 4)
        self.critic_ex = nn.Linear(400, 1)
        self.critic_int = nn.Linear(400, 1)

    def policy(self, input):
        x = input.view(-1, n_stack, 84, 84)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.lin11(x))
        x = F.relu(self.lin12(x))
        return F.softmax(self.actor(x), dim=1)

    def value(self, input):
        x = input.view(-1, n_stack, 84, 84)
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.lin21(x))
        x = F.relu(self.lin22(x))
        ex = self.critic_ex(x)
        # intr = self.critic_int(x)
        return ex #, intr


class RND(nn.Module):
    def __init__(self):
        super(RND, self).__init__()
        self.conv11 = nn.Conv2d(n_stack, 16, 8, stride=4)
        self.conv12 = nn.Conv2d(16, 32, 4, stride=2)
        self.lin11 = nn.Linear(2592, 512)
        self.lin12 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 1)

    def forward(self, input):
        x = input.view(-1, n_stack, 84, 84)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.lin11(x))
        x = F.relu(self.lin12(x))
        return self.out(x)


epochs = 10000
agent = PPO(ActorCritic, num_envs, RND=None)

try:
    agent.model.load_state_dict(torch.load('StateDicts/' + filename + '.pt'))
    # agent.network.load_state_dict(torch.load('StateDicts/' + filename + 'Network.pt'))
    # agent.predictor.load_state_dict(torch.load('StateDicts/' + filename + 'Predictor.pt'))
    print("Files loaded")
except FileNotFoundError:
    print("Files not found")
    pass
except RuntimeError:
    print("New Model")
    pass

avg_rewards = []
rewards_deque = deque(maxlen=100)
rewards = []
episode_loss = []
t = 0

state = np.moveaxis(env.reset(), 3, 1) / 128

for i in range(1, epochs + 1):

    episode_reward = 0
    done = np.array([False for _ in range(num_envs)])

    losses = []

    action = 0
    for _ in range(num_envs):

        while True:
            # cv2.imshow('window', state[0, 0, :, :].reshape(84, 84)*128)
            # if cv2.waitKey(25) == ord('q'):
            #     break
            #
            if i % 100 == 0:
                env.render(mode="human")
            action = agent.choose_action(state)
            # time.sleep(0.01)

            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done, train=True)

            if True in done:
                break
            next_state = np.moveaxis(next_state, 3, 1) / 128
            episode_reward += np.mean(reward)
            state = next_state
            t += 1

            torch.cuda.empty_cache()
    episode_loss.append(np.mean(losses))
    rewards.append(episode_reward)
    rewards_deque.append(episode_reward)
    print('\rEpisode {}, Time {}: {:.3f},  Avg:{:.3f}'.format(i, t, episode_reward, np.mean(rewards_deque)), end='')
    writer.add_scalar('rewards', episode_reward, t)
    if i % 10 == 0:
        avg_rewards.append(np.mean(rewards))
    if i % 100 == 0:
        print(
            '\rEpisode {}, Time {}: {:.3f},  Avg:{:.3f}'.format(i, t, episode_reward, np.mean(rewards_deque)))
        torch.save(agent.model.state_dict(), 'StateDicts/' + filename + '.pt')
        # torch.save(agent.network.state_dict(), 'StateDicts/' + filename + 'Network.pt')
        # torch.save(agent.predictor.state_dict(), 'StateDicts/' + filename + 'Predictor.pt')
        rewards.clear()

torch.save(agent.model.state_dict(), 'StateDicts/' + filename + '.pt')

fig, axs = plt.subplots(2, 1)
axs[0].plot(range(len(avg_rewards)), avg_rewards)
axs[1].plot(range(len(episode_loss)), episode_loss)
plt.show()
