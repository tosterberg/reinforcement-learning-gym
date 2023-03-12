import numpy as np


class Policy:
    """
        Policy class
            interface
                name: overwrite the specified policy name for reporting
            methods
                apply: given estimated rewards, and number of actions available
                    returns the action determined by the policy
                policy_label: returns the name of the policy
    """
    def __init__(self, name='Random'):
        self.name = name

    def __str__(self):
        return str(self.name)

    def apply(self, estimated_values, num_actions):
        return np.random.choice(len(estimated_values))


class GreedyPolicy(Policy):
    """
        GreedyPolicy class extends Policy class
            interface
                initialized: default False, toggles whether the estimated values coming into the policy
                    have been initialized or not. i.e. pull each arm once, or optimize from current
                    values
            variables
                explored: track indexes of arms not yet pulled, -1 when all arms have been pulled
            methods
                policy_label: returns label for identifying the policy
                initial_plays: schedules exploration of arms given uninitialized estimates and explored is not -1
                apply: given estimated rewards, and number of actions available
                    returns the action determined by the greedy policy
    """
    def __init__(self, initialized=False):
        super().__init__(name='Greedy')
        self.initialized = initialized  # Set to true after trying every arm once then greedy
        self.explored = None

    def __str__(self):
        config = 'Pull once' if self.explored is None else 'Agent initialized'
        return f'{self.name}: {config}'

    def initial_plays(self, num_actions):
        if self.explored is None:
            self.explored = num_actions - 1
            return self.explored
        action = self.explored
        self.explored -= 1
        if self.explored == -1:
            self.initialized = True
        return action

    def apply(self, estimated_values, num_actions):
        if not self.initialized:
            return self.initial_plays(num_actions)
        greedy_action = np.argmax(estimated_values)
        action = np.where(estimated_values == np.argmax(estimated_values))[0]
        if len(action) == 0:
            return greedy_action
        return np.random.choice(action)


class EpsilonGreedyPolicy(Policy):
    """
        GreedyPolicy class extends Policy class
        interface
                epsilon: probability that exploratory actions are taken rather than greedy actions
            methods
                apply: given estimated rewards, and number of actions available
                    returns the action determined by the policy
                policy_label: returns the name of the policy
            private methods:
                _greedy: applies greedy search to estimated_values to determine action
                _explore: applies random play to actions to determine action
    """
    def __init__(self, epsilon=0.1):
        self.estimated_values = None
        self.epsilon = epsilon
        super().__init__(name=f'EpsilonGreedy: {self.epsilon:.2f}')

    def __str__(self):
        return super.__str__(self)

    def apply(self, estimated_values, num_actions):
        self.estimated_values = estimated_values
        if np.random.random() < self.epsilon:
            return self._explore()
        else:
            return self._greedy()

    def _greedy(self):
        if self.estimated_values is None:
            raise Exception("Call apply function")
        greedy_action = np.argmax(self.estimated_values)
        action = np.where(self.estimated_values == np.argmax(self.estimated_values))[0]
        if len(action) == 0:
            return greedy_action
        return np.random.choice(action)

    def _explore(self):
        if self.estimated_values is None:
            raise Exception("Call apply function")
        return np.random.choice(len(self.estimated_values))
