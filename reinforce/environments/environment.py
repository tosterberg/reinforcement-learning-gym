import numpy as np
from dataclasses import dataclass


@dataclass
class EnvironmentResult(object):
    """Class for reporting environment runs with a common interface"""
    agent_label: str
    scenario_label: str
    steps: int
    iterations: int
    scores: list
    optimal_actions: list

    def __init__(self, agent_label, scenario_label, steps, iterations, scores, optimal_actions):
        self.agent_label = agent_label
        self.scenario_label = scenario_label
        self.steps = steps
        self.iterations = iterations
        self.scores = scores
        self.optimal_actions = optimal_actions

    def __str__(self):
        output = ''
        output += f'{self.agent_label}\n'
        output += f'{self.scenario_label}\n'
        output += f'Environment: steps={self.steps}, iterations={self.iterations}\n'
        output += f'Total reward: {sum(sum(self.scores)):,.2f}\n'
        return output


class Environment:
    def __init__(self, scenario, agent, steps, iterations, **kwargs):
        self.scenario = scenario
        self.agent = agent
        self.steps = steps
        self.iterations = iterations
        self.result_scores = list()
        self.result_opt = list()
        self.results = None
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._validate()
        self.run()

    def run(self):
        scores = np.zeros(self.steps)
        optimal_scores = np.zeros(self.steps)
        self.setup_tests(scores, optimal_scores)
        self.results = EnvironmentResult(
            agent_label=self.agent.agent_label(),
            scenario_label=self.scenario.scenario_label(),
            steps=self.steps,
            iterations=self.iterations,
            scores=self.result_scores,
            optimal_actions=self.result_opt
        )

    def setup_tests(self, scores, optimal_scores):
        for test in range(self.iterations):
            self.scenario.reset()
            self.agent.reset()
            avg, opt = self.score_agent(scores, optimal_scores)
            self.result_scores.append(avg)
            self.result_opt.append(opt)

    def score_agent(self, scores, optimal_scores):
        for step in range(self.steps):
            action = self.agent.play()
            reward = self.scenario.reward(action)
            self.agent.learn(reward)
            scores[step] += reward
            if action == self.scenario.opt:
                optimal_scores[step] += 1
        avg_score = scores / self.iterations
        opt_score = optimal_scores / self.iterations
        return avg_score, opt_score

    def summarize_test(self):
        print(str(self.results))

    def _validate(self):
        assert (self.scenario is not None)
        assert (self.agent is not None)
        assert (self.iterations > 0)
        assert (self.steps > 0)
