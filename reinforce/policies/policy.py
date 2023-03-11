import numpy as np


class Policy:
    def __init__(self, name='Random'):
        self.name = name

    def policy_label(self):
        return str(self.name)

    def apply(self, estimated_values, num_actions):
        return np.random.choice(len(estimated_values))


class GreedyPolicy(Policy):
    def __init__(self, alt_init_value=None):
        super().__init__(name='Greedy')
        self.initialized = False  # Set to true after trying every arm once then greedy
        self.alt_init_value = alt_init_value
        self. explored = None

    def policy_label(self):
        return str(self.name)

    def initial_plays(self, num_actions):
        if self.explored is None:
            self.explored = num_actions
            return self.explored
        action = self.explored
        self.explored -= 1
        if self.explored == -1:
            self.initialized = True
        return action

    def apply(self, estimated_values, num_actions):
        if not self.initialized:
            return self.initial_plays(num_actions)
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

    def apply(self, estimated_values, num_actions):
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
