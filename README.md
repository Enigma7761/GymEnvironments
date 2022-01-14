#About

This repository was for me messing around with deep reinforcement learning
algorithms, especially trying to implement them from their papers. I have since
abandoned this to make a new one because this one isn't well
structured and I have since realized some of the errors in the implementations.\

Below are some of agents I have properly trained, 
as well as a 2048 gym environment I implemented.

### Finished algorithms

![Bipedal Walker](bipedalwalker.gif)\
Implemented with a DDPG algorithm

![2048](2048.gif)\
Implemented with a DQN algorithm

![Pong](pong.gif)\
Implemented with a DQN algorithm

![Breakout](breakout.gif)\
There are two programs that work for Breakout, one PPO one Rainbow (Rainbow has some
implementation errors)

### Unfinished algorithms

#### DQN + GAN for exploration
Worked on combining a GAN with a DQN algorithm for exploration
in the Summer Stem Institute program. Didn't finish because lacked the
resources to fully evaluate the algorithms performance
and later realized some of the implementation issues in the algorithm itself. \
(MontezumaGAN.py, though it is currently set for Breakout)
(Paper written while in the Summer Stem Institute: https://www.overleaf.com/read/tsxbtnpnjhkf)

#### Overwatch
Was able to make an HP bar reader and an elimination detector, was 
planning to use them for rewards for a reinforcement learning
algorithm but realized the idea was too ambitious for me at the time.