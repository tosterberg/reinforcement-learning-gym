import numpy as np
from .policy import Policy


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
