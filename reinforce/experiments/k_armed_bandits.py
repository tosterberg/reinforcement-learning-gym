from reinforce.scenarios.bandit import Bandit
from reinforce.policies.policy import EpsilonGreedyPolicy
from reinforce.agents.agent import Agent
from reinforce.environments.environment import Environment


if __name__ == '__main__':
    k_arms = 10
    steps = 12000
    runs = 100
    eps = [0, 0.01, 0.1, 0.2, 1]

    for run, epsilon in enumerate(eps):
        scenario = Bandit(arms=k_arms, mean=0, std=1)
        policy = EpsilonGreedyPolicy(epsilon=epsilon)
        agent = Agent(num_actions=k_arms, policy=policy)
        environment = Environment(scenario=scenario, agent=agent, steps=steps, iterations=runs)
        scores, _ = environment.run()
        print(f'Run {run} - Complete')
        print(scenario.scenario_label())
        print(agent.agent_label())
        print(f'{sum(sum(scores)):,.2f}')

