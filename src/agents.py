import random
from src.utils.mol_utils import base_smiles


def greedy_choice(action_space, metric):
    best = (None, float("-inf"))
    for action in action_space:
        score = metric(action)
        if score > best[1]:
            best = (action, score)
    return best[0]


class Agent:

    def rollout(self, env):
        raise NotImplementedError()


class EpsilonGreedyAgent(Agent):

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def rollout(self, env, verbose=False):
        env.reset()

        obs = env.state
        value = 0.0
        for _ in range(env.max_steps):
            action_space = env.valid_actions
            if random.random() < self.epsilon:
                act = random.choice(action_space)
            else:
                act = greedy_choice(action_space, env.prop_fn)
            obs, reward, _ = env.step(act)
            value += reward

            if verbose:
                print(base_smiles(obs[0]), reward)

        return obs[0], value
