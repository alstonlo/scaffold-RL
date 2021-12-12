import random

import numpy as np


# Reference: https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
class ReplayBuffer:

    def __init__(self, size):
        self._storage = []
        self._size = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs, action, reward, next_obs, done):
        data = (obs, action, reward, next_obs, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._size

    def _encode_sample(self, idxs):
        obses, actions, rewards, next_obses, dones = [], [], [], [], []
        for i in idxs:
            data = self._storage[i]
            obs, action, reward, next_obs, done = data
            obses.append(np.array(obs, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_obses.append(np.array(next_obs, copy=False))
            dones.append(done)
        sample = (obses, actions, rewards, next_obses, dones)
        return tuple(np.array(x) for x in sample)

    def sample(self, batch_size):
        idxs = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxs)
