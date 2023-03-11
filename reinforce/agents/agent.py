import numpy as np


class Agent:
    def __init__(self, num_actions, policy, init_value=0, **kwargs):
        self.num_actions = num_actions
        self.policy = policy
        self.step_count = 0
        self.last_action = None
        self.k_actions = np.zeros(self.num_actions)
        self.rewards = np.full(self.num_actions, init_value, dtype=float)
        self.estimated_values = np.zeros(self.num_actions)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def agent_label(self):
        return self.policy.policy_label()

    def learn(self, reward):
        if self.last_action is None:
            raise Exception('Agent must play before learning')
        self.k_actions[self.last_action] += 1
        self.rewards[self.last_action] += reward

        # Update action-value sum(r)/ka
        self.estimated_values[self.last_action] = self.rewards[self.last_action] / self.k_actions[self.last_action]
        self.step_count += 1

    def play(self):
        self.last_action = self.policy.apply(self.estimated_values, self.num_actions)
        return self.last_action

    def summarize(self):
        print(self.agent_label())
        print(sum(self.rewards))
        print(self.k_actions)
        print(self.__dict__)

    def reset(self):
        self.step_count = 0
        self.last_action = None
        self.k_actions = np.zeros(self.num_actions)
        self.rewards = np.zeros(self.num_actions)
        self.estimated_values = np.zeros(self.num_actions)
