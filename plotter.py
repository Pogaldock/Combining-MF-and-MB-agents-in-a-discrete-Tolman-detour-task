import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def plot_reward_rates(reward_rates):
    data = pd.DataFrame()
    for key, value in reward_rates.items():
        if not value:
            continue
        agent_reward_rates = np.array(value)
        df = pd.DataFrame(np.transpose(agent_reward_rates))
        df['episode'] = range(agent_reward_rates.shape[1])
        df = df.melt(id_vars='episode')
        df['agent'] = [key] * len(df)
        data = pd.concat([data, df])

    data = data.rename(columns={'value': 'reward rate'})
    return sns.lineplot(data=data, x='episode', y='reward rate', hue='agent')

def plot_time_vs_reward(reward_rates, timer):

    _, ax = plt.subplots(figsize=(8, 5))

    for key in reward_rates.keys():

        rewards = np.array(reward_rates[key])
        mean_reward_per_run = rewards.mean(axis=1)

        x = np.mean(timer[key])
        y = mean_reward_per_run.mean()
        x_error = np.std(timer[key])
        y_error = mean_reward_per_run.std()

        ax.errorbar(x, y, xerr=x_error, yerr=y_error,
                    fmt='o', markersize=9, capsize=4, label=key)
        ax.text(x, y, '  ' + key, fontsize=9, va='center')

    ax.set_xlabel('Mean computation time (s)')
    ax.set_ylabel('Mean reward rate')
    ax.set_title('Speed / accuracy trade-off')
    ax.grid(True, alpha=0.3)
    ax.legend()

    return ax


def _value_iteration_data(value_iteration_counts):
    data = pd.DataFrame()
    for key, value in value_iteration_counts.items():
        if not value:
            continue
        agent_counts = np.array(value)
        df = pd.DataFrame(np.transpose(agent_counts))
        df['episode'] = range(agent_counts.shape[1])
        df = df.melt(id_vars='episode')
        df['agent'] = [key] * len(df)
        data = pd.concat([data, df])

    return data.rename(columns={'value': 'value iteration calls'})


def plot_value_iteration_calls(value_iteration_counts, obstruct_episode=None):
    data = _value_iteration_data(value_iteration_counts)
    ax = sns.lineplot(data=data, x='episode', y='value iteration calls', hue='agent')
    ax.set_title('Value iteration calls per episode')
    ax.set_ylabel('Mean value iteration calls')

    if obstruct_episode is not None:
        ax.axvline(x=obstruct_episode, linewidth=2, color='black', ls='--')

    return ax


def plot_value_iteration_before_after(value_iteration_counts, obstruct_episode):
    phase_rows = []
    for agent_name, runs in value_iteration_counts.items():
        for run_idx, counts in enumerate(runs):
            before_wall = sum(counts[:obstruct_episode])
            after_wall = sum(counts[obstruct_episode:])
            phase_rows.append({
                'agent': agent_name,
                'run': run_idx,
                'phase': 'before wall',
                'value iteration calls': before_wall
            })
            phase_rows.append({
                'agent': agent_name,
                'run': run_idx,
                'phase': 'after wall',
                'value iteration calls': after_wall
            })

    data = pd.DataFrame(phase_rows)
    _, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=data, x='agent', y='value iteration calls', hue='phase', errorbar='sd', ax=ax)
    ax.set_title('Value iteration calls before and after the wall')
    ax.set_xlabel('agent')
    ax.set_ylabel('Mean value iteration calls')
    ax.tick_params(axis='x', rotation=20)
    ax.grid(True, axis='y', alpha=0.3)

    return ax


def _decision_control_data(decision_control_rates):
    rows = []
    for agent_name, runs in decision_control_rates.items():
        for run_idx, run in enumerate(runs):
            for episode_idx, control_rates in enumerate(run):
                for controller, control_rate in control_rates.items():
                    rows.append({
                        'agent': agent_name,
                        'run': run_idx,
                        'episode': episode_idx,
                        'controller': controller,
                        'control rate': control_rate
                    })

    return pd.DataFrame(rows)


def plot_decision_control_grid(decision_control_rates, obstruct_episode=None):
    data = _decision_control_data(decision_control_rates)
    agents = list(decision_control_rates.keys())
    _, axes = plt.subplots(len(agents), 1, figsize=(10, 3.2 * len(agents)), sharex=True, sharey=True)

    if len(agents) == 1:
        axes = [axes]

    for ax, agent_name in zip(axes, agents):
        agent_data = data[data['agent'] == agent_name]
        sns.lineplot(
            data=agent_data,
            x='episode',
            y='control rate',
            hue='controller',
            errorbar='sd',
            ax=ax
        )
        ax.axhline(y=0.5, linewidth=1.5, color='black', ls=':')
        ax.set_title(agent_name)
        ax.set_ylabel('Control rate')
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)

        if obstruct_episode is not None:
            ax.axvline(x=obstruct_episode, linewidth=2, color='black', ls='--')

    axes[-1].set_xlabel('episode')
    return axes
