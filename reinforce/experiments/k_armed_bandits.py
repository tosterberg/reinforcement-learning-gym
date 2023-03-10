from reinforce.scenarios.stationary_bandit import StationaryBandit
from reinforce.policies.greedy import GreedyPolicy
from reinforce.policies.epsilon_greedy import EpsilonGreedyPolicy
from reinforce.policies.policy import Policy
from reinforce.agents.agent import Agent
from reinforce.environments.environment import Environment


if __name__ == '__main__':
    k = 10
    scenario = StationaryBandit(arms=k, mean=0, std=1)
    policies = [
        GreedyPolicy(),
        EpsilonGreedyPolicy(epsilon=0.2),
        Policy()
    ]
    agents = [
        Agent(num_actions=k, policy=policies[0]),
        Agent(num_actions=k, policy=policies[1]),
        Agent(num_actions=k, policy=policies[2])
    ]
    environment = Environment(scenario=scenario, agents=agents, steps=2000, iterations=100)
    environment.run()
