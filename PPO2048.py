import gym
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch
from collections import deque
import time
from stable_baselines3.common.cmd_util import make_vec_env
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO, A2C
from stable_baselines3.ppo import MlpPolicy
import gym2048


env = make_vec_env("gym2048-v0", n_envs=1)
print(env.reset().shape)

policy_kwargs = dict(net_arch=[dict(vf=[512, 256, 256, 100], pi=[512, 256, 256, 100])])

model = PPO.load("ppo_2048_new")

# model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, n_steps=512, n_epochs=20, batch_size=4096, verbose=2, tensorboard_log="runs", create_eval_env=True, seed=0)
# model.env = env
# model.tensorboard_log = "drive"
# model.learn(total_timesteps=30000000)
# model.save("ppo_2048")

#
obs = env.reset()
scores = []
tiles = []
counter = 0
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if True in dones:
      info_done = np.asarray(info)[dones].tolist()
      for l in range(len(info_done)):
          scores.append(info_done[l]["score"])
          tiles.append(info_done[l]["highest_tile"])
      counter += 1
      if counter >= 12:
        break
    env.render()

fig, axs = plt.subplots(2, 1)
axs[0].plot(range(len(scores)), scores)
axs[0].set_label("scores")
axs[1].plot(range(len(tiles)), tiles)
axs[1].set_label("highest tile")
plt.show()