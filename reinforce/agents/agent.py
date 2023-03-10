import numpy as np


class Agent:
    def __init__(self, num_actions, policy, **kwargs):
        self.num_actions = num_actions
        self.policy = policy
        self.step_count = 0
        self.last_action = None
        self.k_actions = np.zeros(self.num_actions)
        self.rewards = np.zeros(self.num_actions)
        self.estimated_rewards = np.zeros(self.num_actions)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def agent_label(self):
        return self.policy.policy_label()

    def learn(self, reward):
        if self.last_action is None:
            raise Exception('Agent must act before learning')
        self.k_actions[self.last_action] += 1
        self.rewards[self.last_action] += reward

        # Update action-value sum(r)/ka
        self.estimated_rewards[self.last_action] = self.rewards[self.last_action] / self.k_actions[self.last_action]
        self.step_count += 1

    def act(self):
        self.last_action = self.policy.apply(self.estimated_rewards)
        return self.last_action

    def reset(self):
        self.step_count = 0
        self.last_action = None
        self.k_actions = np.zeros(self.num_actions)
        self.rewards = np.zeros(self.num_actions)
        self.estimated_rewards = np.zeros(self.num_actions)
