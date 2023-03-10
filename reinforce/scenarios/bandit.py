import numpy as np


class Bandit:
    """
        Bandit class interface
            arms - number of bandits available in environment for reward
            mean - the mean value of the distribution of each bandit
            std - the standard deviation of the distribution of each bandit
    """
    def __init__(self, arms=0, mean=0, std=1, **kwargs):
        self.arms = arms
        self.mean = mean
        self.std = std
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._validate()
        self.actions = np.zeros(self.arms)

    def _validate(self):
        assert (self.arms > 0)
