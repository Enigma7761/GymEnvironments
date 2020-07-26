import numpy as np
import random
from collections import namedtuple, deque

Replay = namedtuple('Replay', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    def __init__(self, maxlen=1000000, batch_size=64):
        self.buffer = deque(maxlen=maxlen)
        self.batch_size = batch_size

    def store(self, state, action, reward, next_state, done):
        self.buffer.append(Replay(state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self):
        episodes = random.sample(self.buffer, self.batch_size)

        states = np.stack([episode.state for episode in episodes])
        actions = np.stack([episode.action for episode in episodes])
        rewards = np.stack([episode.reward for episode in episodes])
        next_states = np.stack([episode.next_state for episode in episodes])
        dones = np.stack([episode.done for episode in episodes])

        return states, actions, rewards, next_states, dones


class PrioritizedReplayBuffer:
    def __init__(self, maxlen=200000, batch_size=64, alpha=0.5):
        self.maxlen = maxlen
        self.buffer = np.empty((maxlen), dtype=object)
        self.batch_size = batch_size
        self.priorities = np.zeros((maxlen), dtype=np.float32)
        self.priorities[0] = 1
        self.alpha = alpha
        self.index = 0
        self.full = False

    def store(self, state, action, reward, next_state, done):
        self.buffer[self.index] = [state, action, reward, next_state, done]
        self.priorities[self.index] = np.max(self.priorities)
        self.index = (self.index + 1) % self.maxlen
        if self.index == 0:
            self.full = True

    def __len__(self):
        return self.buffer.shape[0]

    def sample(self):
        priorities = self.priorities ** self.alpha
        probs = priorities / np.sum(priorities)
        indices = random.choices(range(self.maxlen), weights=probs.tolist(), k=self.batch_size)
        probs = np.array(self.priorities)[indices]
        episodes = np.array(self.buffer)[indices].tolist()

        states = np.stack([episode[0] for episode in episodes])
        actions = np.stack([episode[1] for episode in episodes])
        rewards = np.stack([episode[2] for episode in episodes])
        next_states = np.stack([episode[3] for episode in episodes])
        dones = np.stack([episode[4] for episode in episodes])

        return states, actions, rewards, next_states, dones, probs, indices

    def store_priorities(self, indices, updated_priorities):
        self.priorities[indices] = updated_priorities


class PrioritizedReplayBufferR2D2:
    def __init__(self, maxlen=1000000, batch_size=64, alpha=0.6, seq_length=80, burn_in=40):
        self.maxlen = maxlen
        self.seq_length = seq_length
        self.buffer = np.empty((maxlen), dtype=object)
        self.batch_size = batch_size
        self.priorities = np.zeros((maxlen), dtype=np.float32)
        self.priorities[0] = 1
        self.alpha = alpha
        self.index = 0
        self.full = False
        self.burn_in = burn_in

    def store(self, sequence):
        self.buffer[self.index] = sequence
        self.priorities[self.index] = np.max(self.priorities)
        self.index = (self.index + 1) % self.maxlen
        if self.index == 0:
            self.full = True

    def __len__(self):
        return self.buffer.shape[0]

    def sample(self):
        priorities = self.priorities ** self.alpha
        probs = priorities / np.sum(priorities)
        indices = random.choices(range(self.maxlen), weights=probs.tolist(), k=self.batch_size)
        probs = np.array(self.priorities)[indices]
        episodes = np.array(self.buffer)[indices].tolist()

        states = np.stack([episode[0] for episode in episodes])
        actions = np.stack([episode[1] for episode in episodes])
        rewards = np.stack([episode[2] for episode in episodes])

        return states, actions, rewards, probs, indices

    def store_priorities(self, indices, updated_priorities):
        self.priorities[indices] = updated_priorities

