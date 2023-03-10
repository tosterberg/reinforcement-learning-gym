import numpy as np


class Policy:
    def __init__(self, name='Random'):
        self.name = name

    def policy_label(self):
        return str(self.name)

    def apply(self, estimated_values):
        return np.random.choice(len(estimated_values))
