import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reinforce.environments.environment import EnvironmentResult

"""
    Utility Functions
        conversions:
            dict -> str (recursive)
"""


def dict_to_string(obj):
    out_str = '{'
    for k, v in obj.__dict__.items():
        if hasattr(v, '__iter__') and not isinstance(v, str):
            v = ", ".join([str(x) for x in v])
        out_str += f'{str(k)}: {str(v)}, '
    out_str = out_str[:-2] + '}'
    return out_str


def plot_env_result(env_results: [EnvironmentResult]):
    plt.rcParams['figure.figsize'] = [15, 5]
    sns.set(style="darkgrid", palette='colorblind')
    for env in env_results:
        description = {
            "steps": [x for x in range(env_results[0].steps)],
            "value": env.mean_reward_per_step,
        }
        pd_data = pd.DataFrame(description)
        ax = sns.scatterplot(x="steps", y="value", data=pd_data, label=env.agent_label, alpha=0.5)
        ax.set(xscale="linear", yscale="linear")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Rewards")
        ax.set_title("Data")
    plt.legend()
    plt.show()

    for env in env_results:
        description = {
            "steps": [x for x in range(env_results[0].steps)],
            "value": env.mean_cumulative_reward_per_step
        }
        pd_data = pd.DataFrame(description)
        ax = sns.lineplot(x="steps", y="value", data=pd_data, label=env.agent_label)
        ax.set(xscale="linear", yscale="linear")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Avg. Rewards")
        ax.set_title("Average Reward")
    plt.legend()
    plt.show()
