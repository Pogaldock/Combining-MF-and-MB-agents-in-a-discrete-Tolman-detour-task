import numpy as np


class Environment:
    """
    The parent class of our two different environments
    """

    def __init__(self):
        """
        This is just a placeholder that the children of this class can fill up.
        The transitions will be a big dictionary, mapping:
            - every state to a dictionary that maps
                - each action to a dictionary that contains
                    - a list of possible arriving states, their corresponding probability distribution and rewards
        """
        self._initial_state = None
        self._current_state = None
        self._transitions = None
        return

    # Getters
    def get_states(self) -> list:
        """
        What are the available states in the environment?
        :return: a list of the available states in the environment
        """
        return list(self._transitions.keys())

    def get_actions(self) -> list:
        """
        What are the available actions in the environment?
        :return: a list of the available actions in the environment
        """
        return list(self._transitions[self._current_state].keys())

    def get_current_state(self) -> int:
        """
        What state is the agent currently occupying?
        :return: the current state
        """
        return self._current_state

    # Actions of the environment
    def take_action(self, action: str) -> dict:
        """
        Takes a single step in the environment
        :param action: the action we choose to take
        :return: a dictionary containing the arrival state, the corresponding reward, and whether this was the end of
            an episode and whether this was a common transition
        """
        if action not in self._transitions[self._current_state].keys():
            raise ValueError(f'Action "{action}" is not an available action.')

        possible_states = self._transitions[self._current_state][action]['state']
        possible_rewards = self._transitions[self._current_state][action]['reward']
        transition_proba = self._transitions[self._current_state][action]['proba']
        random_index = np.random.choice(range(len(possible_states)), p=transition_proba)
        real_transition = True
        if self._current_state == possible_states[random_index]:
            real_transition = False
        self._current_state = possible_states[random_index]
        reward = possible_rewards[random_index]
        end_of_episode = self._current_state == self._initial_state and real_transition
        common = transition_proba[random_index] == max(transition_proba)
        return {'state': self._current_state, 'reward': reward, 'terminated': end_of_episode, 'common': common}


class TolmanMaze(Environment):
    """
    A child to the Environment class.
    This is a discrete instantiation of the Tolman maze following the work of Martinet et al. (2011)
    """

    def __init__(self) -> None:
        """
        Setting up the Tolman Maze.
        :return:
        """
        super().__init__()
        self._initial_state = 0
        self._current_state = self._initial_state
        self._transitions = {0: {'LEFT': {'state': [0], 'proba': [1], 'reward': [0]},
                                 'UP': {'state': [2], 'proba': [1], 'reward': [0]},
                                 'RIGHT': {'state': [0], 'proba': [1], 'reward': [0]}},
                             1: {'LEFT': {'state': [1], 'proba': [1], 'reward': [0]},
                                 'UP': {'state': [6], 'proba': [1], 'reward': [0]},
                                 'RIGHT': {'state': [1], 'proba': [1], 'reward': [0]}},
                             2: {'LEFT': {'state': [1], 'proba': [1], 'reward': [0]},
                                 'UP': {'state': [5], 'proba': [1], 'reward': [0]},
                                 'RIGHT': {'state': [3], 'proba': [1], 'reward': [0]}},
                             3: {'LEFT': {'state': [3], 'proba': [1], 'reward': [0]},
                                 'UP': {'state': [3], 'proba': [1], 'reward': [0]},
                                 'RIGHT': {'state': [4], 'proba': [1], 'reward': [0]}},
                             4: {'LEFT': {'state': [4], 'proba': [1], 'reward': [0]},
                                 'UP': {'state': [9], 'proba': [1], 'reward': [0]},
                                 'RIGHT': {'state': [4], 'proba': [1], 'reward': [0]}},
                             5: {'LEFT': {'state': [5], 'proba': [1], 'reward': [0]},
                                 'UP': {'state': [8], 'proba': [1], 'reward': [0]},
                                 'RIGHT': {'state': [5], 'proba': [1], 'reward': [0]}},
                             6: {'LEFT': {'state': [6], 'proba': [1], 'reward': [0]},
                                 'UP': {'state': [6], 'proba': [1], 'reward': [0]},
                                 'RIGHT': {'state': [7], 'proba': [1], 'reward': [0]}},
                             7: {'LEFT': {'state': [7], 'proba': [1], 'reward': [0]},
                                 'UP': {'state': [7], 'proba': [1], 'reward': [0]},
                                 'RIGHT': {'state': [8], 'proba': [1], 'reward': [0]}},
                             8: {'LEFT': {'state': [8], 'proba': [1], 'reward': [0]},
                                 'UP': {'state': [10], 'proba': [1], 'reward': [0]},
                                 'RIGHT': {'state': [8], 'proba': [1], 'reward': [0]}},
                             9: {'LEFT': {'state': [9], 'proba': [1], 'reward': [0]},
                                 'UP': {'state': [12], 'proba': [1], 'reward': [0]},
                                 'RIGHT': {'state': [9], 'proba': [1], 'reward': [0]}},
                             10: {'LEFT': {'state': [10], 'proba': [1], 'reward': [0]},
                                  'UP': {'state': [13], 'proba': [1], 'reward': [0]},
                                  'RIGHT': {'state': [10], 'proba': [1], 'reward': [0]}},
                             11: {'LEFT': {'state': [10], 'proba': [1], 'reward': [0]},
                                  'UP': {'state': [11], 'proba': [1], 'reward': [0]},
                                  'RIGHT': {'state': [11], 'proba': [1], 'reward': [0]}},
                             12: {'LEFT': {'state': [11], 'proba': [1], 'reward': [0]},
                                  'UP': {'state': [12], 'proba': [1], 'reward': [0]},
                                  'RIGHT': {'state': [12], 'proba': [1], 'reward': [0]}},
                             13: {'LEFT': {'state': [13], 'proba': [1], 'reward': [0]},
                                  'UP': {'state': [self._initial_state], 'proba': [1], 'reward': [10]},
                                  'RIGHT': {'state': [13], 'proba': [1], 'reward': [0]}}}
        return

    def obstruct(self) -> None:
        """
        Obstructs the middle corridor in the Tolman Maze
        """
        self._transitions[2]['UP'] = {'state': [2], 'proba': [1], 'reward': [0]}
        return
