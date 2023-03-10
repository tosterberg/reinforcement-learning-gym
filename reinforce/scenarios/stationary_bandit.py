import numpy as np
from .bandit import Bandit
from reinforce.utils import utils


class StationaryBandit(Bandit):
    """
        StationaryBandit class
            A bandit class implementation with stationary action values; i.e. the distributions
            do not change from step to step
        interface
            arms - number of bandits available in environment for reward
            mean - the mean value of the distribution of each bandit
            std - the standard deviation of the distribution of each bandit
            distribution - function name for the actions distribution setup
        variables
            opt - maximum value in the action array
        methods
            distribution_func - the np.random function for the given distribution
            reset - resets the bandit arms action values to new values
    """
    def __init__(self, distribution='normal', reward_dist='normal', scale=1, **kwargs):
        self.distribution = distribution
        self.reward_dist = reward_dist
        self.scale = scale
        self.label = 'StationaryBandit'
        super().__init__(**kwargs)
        self.distribution_func = getattr(np.random, self.distribution)
        self.reward_func = getattr(np.random, self.reward_dist)
        self.opt = 0
        self.reset()

    def __str__(self):
        return utils.dict_to_string(self)

    def scenario_label(self):
        return str(self.label)

    def reward(self, choice):
        return self.reward_func(self.actions[choice], scale=self.scale)

    def reset(self):
        self.actions = self.distribution_func(self.mean, self.std, self.arms)
        self.opt = np.argmax(self.actions)

    def _validate_kwargs(self):
        assert (hasattr(np.random, self.distribution))
