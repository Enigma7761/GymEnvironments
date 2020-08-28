import gym
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
import numpy as np
from DDPG import DDPG
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
env = gym.make('MountainCarContinuous-v0')


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.lin1 = nn.Linear(3, 16)
        self.lin2 = nn.Linear(16, 20)
        self.lin3 = nn.Linear(20, 1)

    def forward(self, input, action):
        action = action.reshape((action.shape[0], 1))
        x = torch.cat([input, action], 1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.lin1 = nn.Linear(2, 16)
        self.lin2 = nn.Linear(16, 20)
        self.lin3 = nn.Linear(20, 1)

    def forward(self, input):
        x = F.relu(self.lin1(input))
        x = F.relu(self.lin2(x))
        x = torch.tanh(self.lin3(x))
        return x


min = env.action_space.low
max = env.action_space.high
epochs = 5000
agent = DDPG(Actor, Critic, env.action_space) #initial policy parameters and Q-function parameters and replay buffer
#Also does hard update

avg_rewards = []
rewards = []
t = 0
for i in range(epochs): #repeat
    state = env.reset() #observe state s
    done = False
    episodes = []
    episode_reward = 0
#    state = state.reshape((3, 96, 96))
#     agent.decrement_std()
    while not done:
        if i % 100 == 0:
            env.render()
        action = agent.choose_action(state, i) #select action a as action+noise
        action = np.clip(action, min, max) #clip it
        next_state, reward, done, _ = env.step(action) #Observe next state, reward, and done
        episode_reward += reward
        agent.store(state, action, reward, next_state, done) #Store into replay buffer
        state = next_state
        t += 1
        if t % 8 == 0: #if time to update
            agent.train()

    rewards.append(episode_reward)
    if i % 100 == 0 and i > 0:
        print('Episode {}: {}, {}'.format(i, np.mean(rewards), agent.noise.sigma))
        avg_rewards.append(np.mean(rewards))
        if np.mean(rewards) >= 110:
            break
        rewards.clear()

torch.save(agent.actor.state_dict(), 'StateDicts/mtCarActor.pt')
torch.save(agent.critic.state_dict(), 'StateDicts/mtCar.pt')

plt.plot(range(len(avg_rewards)), avg_rewards)
plt.show()
