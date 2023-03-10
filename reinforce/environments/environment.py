import numpy as np


class Environment:
    def __init__(self, scenario, agents, steps, iterations, **kwargs):
        self.scenario = scenario
        self.agents = agents
        self.steps = steps
        self.iterations = iterations
        self.result_scores = list()
        self.result_opt = list()
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._validate()

    def run(self):
        scores = np.zeros((self.steps, len(self.agents)))
        optimal_scores = np.zeros((self.steps, len(self.agents)))
        self.setup_tests(scores, optimal_scores)
        print(self.result_scores)
        print(self.result_opt)

    def setup_tests(self, scores, optimal_scores):
        for test in range(self.iterations):
            print(f'Test: {test}')
            self.scenario.reset()
            for agent in self.agents:
                agent.reset()
                avg, opt = self.score_agents(scores, optimal_scores)
                self.result_scores.append(avg)
                self.result_opt.append(opt)

    def score_agents(self, scores, optimal_scores):
        for step in range(self.steps):
            for idx, agent in enumerate(self.agents):
                action = agent.act()
                reward = self.scenario.reward(action)
                agent.learn(reward)
                scores[step, idx] += reward
                if action == self.scenario.opt:
                    optimal_scores[step, idx] += 1
        avg_score = scores / self.iterations
        opt_score = optimal_scores / self.iterations
        return avg_score, opt_score

    def _validate(self):
        assert (self.scenario is not None)
        assert (len(self.agents) > 0)
        assert (self.iterations > 0)
        assert (self.steps > 0)
