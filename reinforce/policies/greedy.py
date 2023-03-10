import numpy as np
from .policy import Policy


class GreedyPolicy(Policy):
    def __init__(self):
        super().__init__(name='Greedy')

    def policy_label(self):
        return str(self.name)

    def apply(self, estimated_rewards):
        greedy_action = np.argmax(estimated_rewards)
        action = np.where(estimated_rewards == np.argmax(estimated_rewards))[0]
        if len(action) == 0:
            return greedy_action
        return np.random.choice(action)
