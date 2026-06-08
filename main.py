from experiment import *
from combined_agent2 import Adaptive, Baseline, Dissagreement
from plotter import *
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def main():
    
    # The experiment ##################################################################################################
    
    exp_params = {}
    exp_params['nr_of_episodes'] = 200  # ----------- How many episodes we want to model
    exp_params['nr_of_runs'] = 200 # ----------------- How many times do we want to run the entire experiment
    exp_params['obstruct_corridor'] = 100 # ------- In which episode should we block the middle corridor (int or None)

    # The MF agent #####################################################################################################
    
    MF_agent_params = {}
    MF_agent_params['gamma'] = 0.9  # ---------------- The discount factor
    MF_agent_params['epsilon'] = 0.05  # ------------- For the epsilon-greedy action selection
    MF_agent_params['alpha'] = 0.3  # ---------------- The learning rate of the MF agent

    # The MB agent #####################################################################################################
    
    MB_agent_params = {}
    MB_agent_params['gamma'] = 0.9  # ---------------- The discount factor
    MB_agent_params['epsilon'] = 0.05  # ------------- For the epsilon-greedy action selection
    MB_agent_params['theta'] = 0.1  # ---------------- The threshold of the MB agent
    MB_agent_params['window_length'] = 10  # --------- The window for the model learning (MB)

    # The combined agent ###############################################################################################
    
    combined_agent_params = {}
    combined_agent_params['gamma'] = 0.9  # ----------- The discount factor
    combined_agent_params['epsilon'] = 0.05  # -------- For the epsilon-greedy action selection
    combined_agent_params['alpha'] = 0.3  # ----------- The learning rate of the MF agent
    combined_agent_params['theta'] = 0.1  # ----------- The threshold of the MB agent
    combined_agent_params['window_length'] = 10  # ---- The window for the model learning (MB)
    combined_agent_params['threshold'] = 0.02  # ------- The threshold for allowing the MB agent to learn

    combined_agents = {
        'baseline': Baseline,
        'adaptive': Adaptive,
        'dissagreement': Dissagreement
    }

    # Running the actual experiment ####################################################################################
    
    reward_rates = {agent_name: [] for agent_name in combined_agents.keys()}
    times = {agent_name: [] for agent_name in combined_agents.keys()}
    value_iteration_counts = {agent_name: [] for agent_name in combined_agents.keys()}
    decision_control_rates = {agent_name: [] for agent_name in combined_agents.keys()}
    
    for _ in tqdm(range(exp_params['nr_of_runs'])):
        
        # # --------------------------- The MF agent ---------------------------------------------------------------------
        # environment = TolmanMaze()
        # MF_agent_params['environment'] = environment
        # MF_agent = MFagent(**MF_agent_params)
        # start = time.time()
        # MF_reward_rate = run_experiment(exp_params=exp_params, environment=environment, agent=MF_agent)
        # elapsed = time.time() - start
        # times['MF'].append(elapsed)
        # print(f"{exp_params['nr_of_episodes']} episodes finished in {elapsed} sec by the MF agent.")
        # reward_rates['MF'].append(MF_reward_rate)

        # # --------------------------- The MB agent ---------------------------------------------------------------------
        # environment = TolmanMaze()
        # MB_agent_params['environment'] = environment
        # MB_agent = MBagent(**MB_agent_params)
        # start = time.time()
        # MB_reward_rate = run_experiment(exp_params=exp_params, environment=environment, agent=MB_agent)
        # elapsed = time.time() - start
        # times['MB'].append(elapsed)
        # print(f"{exp_params['nr_of_episodes']} finished in {elapsed} sec by the MB agent.")
        # reward_rates['MB'].append(MB_reward_rate)

        # --------------------------- The combined agents ---------------------------------------------------------------
        for agent_name, agent_class in combined_agents.items():
            environment = TolmanMaze()
            agent_params = combined_agent_params.copy()
            agent_params['environment'] = environment
            agent = agent_class(**agent_params)

            start = time.time()
            reward_rate = run_experiment(exp_params=exp_params,
                                         environment=environment,
                                         agent=agent)
            elapsed = time.time() - start

            times[agent_name].append(elapsed)
            reward_rates[agent_name].append(reward_rate)
            value_iteration_counts[agent_name].append(agent.value_iteration_history)
            decision_control_rates[agent_name].append(agent.decision_control_history)

            wall_episode = exp_params['obstruct_corridor']
            before_wall_calls = sum(agent.value_iteration_history[:wall_episode])
            after_wall_calls = sum(agent.value_iteration_history[wall_episode:])
            # print(
            #     f"{exp_params['nr_of_episodes']} finished in {elapsed} sec by {agent_name}. "
            #     f"value_iteration calls: total={agent.value_iteration_calls}, "
            #     f"before wall={before_wall_calls}, after wall={after_wall_calls}."
            # )
            
    # Plotting #########################################################################################################
    
    ax = plot_reward_rates(reward_rates)
    ax.axhline(y=10 / 6, linewidth=2, color='black', ls='--')
    ax.axhline(y=10 / 8, linewidth=2, color='black', ls='--')
    plt.show()
    
    # Plot time vs reward precision for all models
    plot_time_vs_reward(reward_rates, times)
    plt.show()

    plot_value_iteration_calls(value_iteration_counts, exp_params['obstruct_corridor'])
    plt.savefig('figures/value_iteration_calls_per_episode.png', dpi=300, bbox_inches='tight')
    plt.show()

    plot_value_iteration_before_after(value_iteration_counts, exp_params['obstruct_corridor'])
    plt.savefig('figures/value_iteration_calls_before_after_wall.png', dpi=300, bbox_inches='tight')
    plt.show()

    plot_decision_control(decision_control_rates, exp_params['obstruct_corridor'])
    plt.savefig('figures/decision_control_per_episode.png', dpi=300, bbox_inches='tight')
    plt.show()

    plot_mb_decision_control(decision_control_rates, exp_params['obstruct_corridor'])
    plt.savefig('figures/mb_decision_control_per_episode.png', dpi=300, bbox_inches='tight')
    plt.show()

    plot_decision_control_grid(decision_control_rates, exp_params['obstruct_corridor'])
    plt.savefig('figures/decision_control_grid_per_agent.png', dpi=300, bbox_inches='tight')
    plt.show()

    plot_mb_decision_control_heatmap(decision_control_rates, exp_params['obstruct_corridor'])
    plt.savefig('figures/mb_decision_control_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    print()
    for agent_name in combined_agents.keys():
        print(f"Average reward for {agent_name}: ", np.mean(reward_rates[agent_name]))

    print()
    for agent_name in combined_agents.keys():
        print(f"Average time for {agent_name}: ", np.mean(times[agent_name]), "sec")

    print()
    for agent_name in combined_agents.keys():
        counts = np.array(value_iteration_counts[agent_name])
        wall_episode = exp_params['obstruct_corridor']
        total_calls = counts.sum(axis=1)
        before_wall_calls = counts[:, :wall_episode].sum(axis=1)
        after_wall_calls = counts[:, wall_episode:].sum(axis=1)
        print(f"Average value_iteration calls for {agent_name}: ", np.mean(total_calls))
        print(f"  before wall: {np.mean(before_wall_calls)}")
        print(f"  after wall: {np.mean(after_wall_calls)}")

    print()
    for agent_name in combined_agents.keys():
        runs = decision_control_rates[agent_name]
        mf_control = np.array([[episode['MF'] for episode in run] for run in runs])
        mb_control = np.array([[episode['MB'] for episode in run] for run in runs])
        print(f"Average MF control for {agent_name}: ", np.mean(mf_control))
        print(f"Average MB control for {agent_name}: ", np.mean(mb_control))


if __name__ == '__main__':
    main()
