import random


# Reference: https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
class ReplayBuffer:

    def __init__(self, size):
        self._storage = []
        self._size = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, sa_t, reward, sa_tp1s, done):
        data = (sa_t, reward, sa_tp1s, done)
        # (s,a) pair, reward, and (s',a') pairs where s' ~ p(s'|s,a)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._size

    def sample(self, batch_size):
        k = min(batch_size, len(self))
        idxs = [random.randrange(len(self._storage)) for _ in range(k)]
        batch = []
        for i in idxs:
            batch.append(self._storage[i])
        return list(zip(*batch))
