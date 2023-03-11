import numpy as np


class Agent:
    def __init__(self, num_actions, policy, init_value=0, **kwargs):
        self.num_actions = num_actions
        self.init_value = init_value
        self.policy = policy
        self.last_action = None
        self.action_counts = np.zeros(self.num_actions)
        self.actions_total_rewards = np.zeros(self.num_actions)
        self.estimated_action_values = np.full(self.num_actions, self.init_value, dtype=float)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def agent_label(self):
        return f'{self.policy.policy_label()}, init_values={self.init_value}'

    def learn(self, reward):
        if self.last_action is None:
            raise Exception('Agent must play before learning')
        self.action_counts[self.last_action] += 1
        self.actions_total_rewards[self.last_action] += reward

        # Update action-value sum(r)/ka
        self.estimated_action_values[self.last_action] = self.actions_total_rewards[self.last_action] / self.action_counts[self.last_action]

    def play(self):
        self.last_action = self.policy.apply(self.estimated_action_values, self.num_actions)
        return self.last_action

    def reset(self):
        self.last_action = None
        self.action_counts = np.zeros(self.num_actions)
        self.actions_total_rewards = np.zeros(self.num_actions)
        self.estimated_action_values = np.full(self.num_actions, self.init_value, dtype=float)
