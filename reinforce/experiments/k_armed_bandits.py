from reinforce.scenarios.bandit import Bandit
from reinforce.policies.policy import EpsilonGreedyPolicy, GreedyPolicy, Policy
from reinforce.agents.agent import Agent
from reinforce.environments.environment import Environment


if __name__ == '__main__':
    # Initial config
    k_arms = 10
    steps = 2000
    runs = 100

    # Test epsilon-greedy
    eps = [0.01, 0.1, 0.2]
    for run, epsilon in enumerate(eps):
        scenario = Bandit(arms=k_arms, mean=0, std=1)
        policy = EpsilonGreedyPolicy(epsilon=epsilon)
        agent = Agent(num_actions=k_arms, policy=policy)
        environment = Environment(scenario=scenario, agent=agent, steps=steps, iterations=runs)
        environment.run()
        print(f'Epsilon-Greedy Run {run+1} - Complete')
        environment.summarize_test()

    # Test alternative greedy
    agents = [
        Agent(num_actions=k_arms, policy=GreedyPolicy()),
        Agent(num_actions=k_arms, policy=GreedyPolicy(initialized=True)),
        Agent(num_actions=k_arms, policy=GreedyPolicy(), init_value=5)
    ]
    for run, agent in enumerate(agents):
        scenario = Bandit(arms=k_arms, mean=0, std=1)
        environment = Environment(scenario=scenario, agent=agent, steps=steps, iterations=runs)
        environment.run()
        print(f'Greedy Run {run+1} - Complete')
        environment.summarize_test()

    # Test random selection policy
    scenario = Bandit(arms=k_arms, mean=0, std=1)
    agent = Agent(num_actions=k_arms, policy=Policy())
    environment = Environment(scenario=scenario, agent=agent, steps=steps, iterations=runs)
    environment.run()
    print(f'Random Run 1 - Complete')
    environment.summarize_test()
