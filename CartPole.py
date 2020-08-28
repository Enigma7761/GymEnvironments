import gym
import numpy as np
from DQN import DQN
import torch
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

env = gym.make('CartPole-v0')
done = False
dqn = DQN(4, 2, device, train=False)
dqn.policy.load_state_dict(torch.load('model.pt'))
dqn.policy.eval()
for i in range(100):
    state = env.reset()
    while not done:
        time.sleep(0.05)
        env.render()
        action = dqn.get_action(state)
        state, _, done, _ = env.step(action)

env.close()
