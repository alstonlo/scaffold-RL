import random


class Agent:

    def sample_action(self, env):
        raise NotImplementedError()

    def rollout(self, env):
        env.reset()

        obs = env.state
        value = 0.0
        for _ in range(env.max_steps):
            act = self.sample_action(env)
            obs, reward, _ = env.step(act)
            value += reward
        return obs[0], value


class EpsilonGreedyAgent(Agent):

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def sample_action(self, env):
        action_space = env.valid_actions

        if random.random() < self.epsilon:
            return random.choice(action_space)
        else:
            best = (None, float("-inf"))
            for action in action_space:
                score = env.prop_fn(action)
                if score > best[1]:
                    best = (action, score)
            return best[0]
