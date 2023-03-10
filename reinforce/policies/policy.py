import numpy as np


class Policy:
    def __init__(self, name='Random'):
        self.name = name

    def policy_label(self):
        return str(self.name)

    def apply(self, estimated_values):
        return np.random.choice(len(estimated_values))


class GreedyPolicy(Policy):
    def __init__(self):
        super().__init__(name='Greedy')

    def policy_label(self):
        return str(self.name)

    def apply(self, estimated_values):
        greedy_action = np.argmax(estimated_values)
        action = np.where(estimated_values == np.argmax(estimated_values))[0]
        if len(action) == 0:
            return greedy_action
        return np.random.choice(action)


class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon=0.1):
        self.estimated_values = None
        self.epsilon = epsilon
        super().__init__(name=f'EpsilonGreedy: {self.epsilon:.2f}')

    def policy_label(self):
        return str(self.name)

    def apply(self, estimated_values):
        self.estimated_values = estimated_values
        if np.random.random() < self.epsilon:
            return self.explore()
        else:
            return self.greedy()

    def greedy(self):
        if self.estimated_values is None:
            raise Exception("Call apply function")
        greedy_action = np.argmax(self.estimated_values)
        action = np.where(self.estimated_values == np.argmax(self.estimated_values))[0]
        if len(action) == 0:
            return greedy_action
        return np.random.choice(action)

    def explore(self):
        if self.estimated_values is None:
            raise Exception("Call apply function")
        return np.random.choice(len(self.estimated_values))
