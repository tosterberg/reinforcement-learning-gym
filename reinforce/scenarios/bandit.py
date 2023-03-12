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
        self.distribution_func = getattr(np.random, self.distribution) \
            if hasattr(np.random, self.distribution) else \
            lambda x, y, z: self.distribution

        self.reward_func = getattr(np.random, self.reward_dist) \
            if hasattr(np.random, self.reward_dist) else \
            lambda x, scale: self.reward_dist
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


class ProvidedBandit:
    """
    ProvidedBandit class is copied from https://github.com/doug57/MSDS684_1 as part of
    MSDS 684 - Regis University, @author: Douglas Hart
    Licensed under the Apache-2.0 license
    """
    def __init__(self, mu, sigma):

        self.mu = mu  # mean of rv's
        self.sigma = sigma  # sdev of rv's
        self.n = 0  # number of rv's generated, incr
        self.xn = 0  # rv from generator
        self.mean = 0  # sample mean
        self.variance = 0  # sample variance
        self.sample_variance = 0  # sample sample variance -
        self.existing_aggregate = [0, 0, 0]  # storage for Welford's algorithm

    def play(self):  # method returns result from play
        self.n += 1
        self.xn = random.normalvariate(self.mu, self.sigma)
        self.existing_aggregate = self.update(self.existing_aggregate, self.xn)
        return self.xn

    def get_statistics(self):
        return self.finalize(self.existing_aggregate)

    # From https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

    # for a new value newValue, compute the new count, new mean, the new M2.
    # mean accumulates the mean of the entire dataset
    # M2 aggregates the squared distance from the mean
    # count aggregates the number of samples seen so far
    def update(self, existing_aggregate, new_value):
        (count, mean, m2) = existing_aggregate
        count += 1
        delta = new_value - mean
        mean += delta / count
        delta2 = new_value - mean
        m2 += delta * delta2

        return count, mean, m2

    # retrieve the mean, variance and sample variance from an aggregate
    def finalize(self, existing_aggregate):
        (count, mean, m2) = existing_aggregate
        (mean, variance, sample_variance) = (mean, m2 / count, m2 / (count - 1))
        if count < 2:
            return float('nan')
        else:
            return mean, variance, sample_variance
