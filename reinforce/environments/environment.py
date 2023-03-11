import numpy as np


class Environment:
    def __init__(self, scenario, agent, steps, iterations, **kwargs):
        self.scenario = scenario
        self.agent = agent
        self.steps = steps
        self.iterations = iterations
        self.result_scores = list()
        self.result_opt = list()
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._validate()

    def run(self):
        scores = np.zeros(self.steps)
        optimal_scores = np.zeros(self.steps)
        self.setup_tests(scores, optimal_scores)
        return self.result_scores, self.result_opt

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

    def _validate(self):
        assert (self.scenario is not None)
        assert (self.agent is not None)
        assert (self.iterations > 0)
        assert (self.steps > 0)
