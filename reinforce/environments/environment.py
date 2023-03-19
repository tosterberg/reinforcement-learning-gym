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

    def __init__(self, agent_label, scenario_label, steps, iterations, scores):
        self.agent_label = agent_label
        self.scenario_label = scenario_label
        self.steps = steps
        self.iterations = iterations
        self.scores = scores
        self.mean_reward_per_step = None
        self.mean_cumulative_reward_per_step = None
        self.calculate_metrics()

    def __str__(self):
        output = ''
        output += f'{self.agent_label}\n'
        output += f'{self.scenario_label}\n'
        output += f'Environment: steps={self.steps}, iterations={self.iterations}\n'
        output += f'Total reward: {sum(sum(self.scores)):,.2f}\n'
        output += f'Mean reward per step: {sum(self.mean_reward_per_step) / self.steps:.4f}\n'
        return output

    def calculate_metrics(self):
        self.mean_reward_per_step = np.zeros(self.steps)
        for score in self.scores:
            for i in range(len(score)):
                self.mean_reward_per_step[i] += score[i]

        for i in range(len(self.mean_reward_per_step)):
            self.mean_reward_per_step[i] /= self.iterations

        accumulator = 0
        self.mean_cumulative_reward_per_step = np.zeros(self.steps)
        for i in range(len(self.mean_reward_per_step)):
            self.mean_cumulative_reward_per_step[i] = (self.mean_reward_per_step[i] + accumulator) / (i + 1)
            accumulator += self.mean_reward_per_step[i]


class Environment:
    """
        Environment class
            interface
                scenario: scenarios provide the number of actions, and their rewards'
                agent: agents interact with a scenario in an environment
                steps: number of interactions agents get with scenario
                iterations: number of times the scenario and agent repeat their steps
            variables
                result_scores: list of scores from the environment run
                result_opt: list of counts where the agent chose optimally
                results: summary object detailing the environment run
            methods:
                run: executes the simulation given the environmental config
                setup_tests: resets agent, and scenario then records the result
                score_agent: iterates interactions between scenario and agent for given steps
                    recording the rewards and optimal plays
                summarize_test: prints the results of the environmental run
    """

    def __init__(self, scenario, agent, steps, iterations, **kwargs):
        self.scenario = scenario
        self.agent = agent
        self.steps = steps
        self.iterations = iterations
        self.result_scores = list()
        self.results = None
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._validate()
        self.run()

    def run(self):
        scores = np.zeros(self.steps)
        self.setup_tests(scores)
        self.results = EnvironmentResult(
            agent_label=str(self.agent),
            scenario_label=str(self.scenario),
            steps=self.steps,
            iterations=self.iterations,
            scores=self.result_scores
        )

    def setup_tests(self, scores):
        for test in range(self.iterations):
            self.scenario.reset()
            self.agent.reset()
            avg = self.score_agent(scores)
            self.result_scores.append(avg)

    def score_agent(self, scores):
        for step in range(self.steps):
            action = self.agent.play()
            reward = self.scenario.reward(action)
            self.agent.learn(reward)
            scores[step] += reward
        avg_score = scores / self.iterations
        return avg_score

    def summarize_test(self):
        print(str(self.results))

    def _validate(self):
        assert (self.scenario is not None)
        assert (self.agent is not None)
        assert (self.iterations > 0)
        assert (self.steps > 0)
