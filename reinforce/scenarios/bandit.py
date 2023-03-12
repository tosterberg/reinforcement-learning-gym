import numpy as np


class Bandit:
    """
        StationaryBandit class
            A bandit class implementation with stationary action values; i.e. the distributions
            do not change from step to step
        interface
            arms - number of bandits available in environment for reward
            mean - the mean value of the distribution of each bandit
            std - the standard deviation of the distribution of each bandit
            seed - random seed to set the random functions with
            distribution - function name for the actions distribution setup
            reward_dist - function name for the rewards distribution setup
            scale - scaling value for the reward_dist function
        variables
            label - scenario name
            opt - maximum value action index in the action array
        methods
            distribution_func - the np.random function for the given distribution
            reward_func - the np.random function for the given distribution
            reset - resets the bandit arms action values to new values
            reward - return reward value given action choice
    """
    def __init__(self, arms=0, mean=0, std=1, seed=42,
                 distribution='normal', reward_dist='normal', scale=1, **kwargs):
        self.arms = arms
        self.mean = mean
        self.std = std
        self.seed = seed
        np.random.seed(self.seed)
        self.distribution = distribution
        self.reward_dist = reward_dist
        self.scale = scale
        self.label = 'Bandit'
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._validate()
        self.actions = np.zeros(self.arms)
        self.distribution_func = getattr(np.random, self.distribution)
        self.reward_func = getattr(np.random, self.reward_dist)
        self.opt = 0
        self.reset()

    def __str__(self):
        return f'{self.label}: arms={self.arms}, mean={self.mean}, std={self.std}'

    def reward(self, action):
        return self.reward_func(self.actions[action], scale=self.scale)

    def reset(self):
        self.actions = self.distribution_func(self.mean, self.std, self.arms)
        self.opt = np.argmax(self.actions)

    def _validate(self):
        assert (self.arms > 0)
        assert (hasattr(np.random, self.distribution))
