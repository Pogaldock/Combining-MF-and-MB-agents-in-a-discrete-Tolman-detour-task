from agents import *


def run_experiment(exp_params: dict, environment: Environment, agent) -> dict:
    """
    Simply runs the RL experiment as specified by the parameters.
    :param exp_params: further parameters of the experiment
    :param: environment: the environment we're using
    :param agent: the agent we're using
    :return: a list containing the reward rates and a
    """
    # Running the experiment ###########################################################################################
    reward_rate = [0] * exp_params['nr_of_episodes']
    for episode_idx in range(exp_params['nr_of_episodes']):
        if hasattr(agent, 'start_episode'):
            agent.start_episode()

        terminated = False
        nr_of_steps = 0

        if exp_params['obstruct_corridor'] is not None and exp_params['obstruct_corridor'] == episode_idx:
            environment.obstruct()

        while not terminated:
            # 1) Check to see what we can do and pick an action to follow
            action = agent.action_selection()
            action = action['action']

            # 2) Take the step
            transition = environment.take_action(action=action)
            terminated = transition['terminated']

            # 3) ... and learn from it
            agent.reinforcement_learning(action=action, arrival_state=transition['state'], reward=transition['reward'])

            nr_of_steps += 1
        reward_rate[episode_idx] = transition['reward'] / nr_of_steps

        if hasattr(agent, 'end_episode'):
            agent.end_episode()
    return reward_rate
