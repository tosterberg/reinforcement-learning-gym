from reinforce.scenarios.bandit import Bandit
from reinforce.policies.policy import EpsilonGreedyPolicy, GreedyPolicy, Policy
from reinforce.agents.agent import Agent
from reinforce.environments.environment import Environment
from reinforce.utils.utils import plot_env_result


if __name__ == '__main__':
    # Initial config
    k_arms = 10
    steps = 2000
    runs = 100

    scenario = Bandit(arms=k_arms, mean=0, std=1)
    results = []

    # Test epsilon-greedy
    eps = [0.01, 0.1, 0.2]
    for epsilon in eps:
        agent = Agent(num_actions=k_arms, policy=EpsilonGreedyPolicy(epsilon=epsilon))
        environment = Environment(scenario=scenario, agent=agent, steps=steps, iterations=runs)
        environment.summarize_test()
        results.append(environment.results)

    # Test alternative greedy
    agents = [
        Agent(num_actions=k_arms, policy=GreedyPolicy()),
        Agent(num_actions=k_arms, policy=GreedyPolicy(initialized=True)),
        Agent(num_actions=k_arms, policy=GreedyPolicy(), init_value=5)
    ]
    for agent in agents:
        environment = Environment(scenario=scenario, agent=agent, steps=steps, iterations=runs)
        environment.summarize_test()
        results.append(environment.results)

    # Test random selection policy
    agent = Agent(num_actions=k_arms, policy=Policy())
    environment = Environment(scenario=scenario, agent=agent, steps=steps, iterations=runs)
    environment.summarize_test()
    results.append(environment.results)

    plot_env_result(results)
