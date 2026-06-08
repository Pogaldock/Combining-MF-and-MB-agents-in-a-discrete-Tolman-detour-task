from environments import *


class RLagent:
    """
    This is the parent class of all agents.
    """

    def __init__(self, environment: Environment, gamma=0.9, epsilon=0.) -> None:
        """
        Setting up the agent. The Q-table will be a dictinoary mapping states to dictionaries, where each dict maps
        actions to their corresponding Q-values
        :return:
        """
        self._gamma = gamma
        self._epsilon = epsilon
        self._Q = {state: {action: 0 for action in environment.get_actions()}
                   for state in environment.get_states()}
        self._current_state = environment.get_current_state()
        return

    # Private methods
    def __take_step__(self, arrival_state: int) -> None:
        """
        Updates the current state in the agent
        """
        self._current_state = arrival_state
        return

    def __Q_max__(self, state: int) -> dict:
        """
        Returns the best Q-value and the corresponding action in a given state
        :param state: the state of interest
        :return: a dict containing both the Q-value and the corresponding actions (list)
        """
        Q_max = {'value': 0, 'action': [list(self._Q[state].keys())[0]]}
        for action in self._Q[state].keys():
            if self._Q[state][action] > Q_max['value']:
                Q_max['value'] = self._Q[state][action]
                Q_max['action'] = [action]
            elif self._Q[state][action] == Q_max['value']:
                Q_max['action'].append(action)
        return Q_max

    # Public methods to instruct the agent
    def action_selection(self) -> dict:
        """
        Chooses what action to take based on the state the agent is in
        :return: the chosen action
        """
        if np.random.uniform(0, 1, 1) <= self._epsilon:
            random_index = np.random.choice(range(len(self._Q[self._current_state].keys())))
            action = list(self._Q[self._current_state].keys())[random_index]
            return {'action': action, 'Q-value': self._Q[self._current_state][action]}
        greedy_actions = self.__Q_max__(state=self._current_state)['action']
        action = np.random.choice(greedy_actions)
        return {'action': action, 'Q-value': self._Q[self._current_state][action]}

    def reinforcement_learning(self, action: str, arrival_state: int, reward: float) -> None:
        """
        This is a placeholder function for all the learning algorithms down the line.
        :param action: the action taken
        :param arrival_state: the state the agent arrived in after taking the action
        :param reward: the reward received over the state transition
        """
        pass


class MFagent(RLagent):
    """
    This is the model-free agent that will use the Q-learning algorithm
    """

    def __init__(self, environment: Environment, gamma=0.9, epsilon=0.05, alpha=0.3):
        """
        Setting up the MF agent. Unlike its parent (and little brother, the MB agent) this agent needs an alpha param
        """
        super().__init__(environment=environment, gamma=gamma, epsilon=epsilon)
        self._alpha = alpha
        return

    # The hidden methods necessary for Q-learning
    def Q_learning(self, action: str, arrival_state: int, reward: float) -> float:
        """
        The Q-learning algorithm
        """
        Q_max = self.__Q_max__(state=arrival_state)['value']
        ################################################################################################################
        TD_error = reward + (self._gamma * Q_max - self._Q[self._current_state][action])
        ################################################################################################################
        self._Q[self._current_state][action] += self._alpha * TD_error
        self.__take_step__(arrival_state=arrival_state)
        return TD_error

    # We overwrite the RL method coming from the parent class
    def reinforcement_learning(self, action: str, arrival_state: int, reward: float) -> None:
        """
        The MF agent will simply use Q-learning to update its Q-values
        :param action: the action taken
        :param arrival_state: the state the agent arrived in after taking the action
        :param reward: the reward received over the state transition
        """
        self.Q_learning(action=action, arrival_state=arrival_state, reward=reward)
        super().reinforcement_learning(action=action, arrival_state=arrival_state, reward=reward)
        return


class MBagent(RLagent):
    """
    This is the model-based agent that will use the VI algorithm
    """

    def __init__(self, environment: Environment, gamma=0.9, epsilon=0.05, theta=0.0001, window_length=10) -> None:
        """
        Setting up the MF agent. This agent will also store a model of the environment in the form of a reward- and a
        transition function. This means that this agent will need a convergence threshold for its calculations (theta)
        and a window length for updating its model
        """
        self._theta = theta
        self._window_length = window_length
        super().__init__(environment=environment, gamma=gamma, epsilon=epsilon)
        self._reward_history = {state: {action: [] for action in environment.get_actions()}
                                for state in environment.get_states()}
        self._reward_fun = {state: {action: 0 for action in environment.get_actions()}
                            for state in environment.get_states()}
        self._transition_history = {state: {action: []
                                            for action in environment.get_actions()}
                                    for state in environment.get_states()}
        self._transition_function = {state: {action: [1 / len(environment.get_states())] * len(environment.get_states())
                                             for action in environment.get_actions()}
                                     for state in environment.get_states()}
        return

    # The hidden methods related to model-based learning
    def update_model(self, action: str, arrival_state: int, reward: float) -> None:
        """
        Updates the world model based on the experienced transition
        :param action: the action taken
        :param arrival_state: the state the agent arrived in after taking the action
        :param reward: the reward received over the state transition
        """
        self._reward_history[self._current_state][action].append(reward)
        if len(self._reward_history[self._current_state][action]) > self._window_length:
            self._reward_history[self._current_state][action].pop(0)
        self._reward_fun[self._current_state][action] = np.average(self._reward_history[self._current_state][action])

        transition = [0] * len(self._Q.keys())
        transition[arrival_state] = 1
        self._transition_history[self._current_state][action].append(transition)
        if len(self._transition_history[self._current_state][action]) > self._window_length:
            self._transition_history[self._current_state][action].pop(0)
        self._transition_function[self._current_state][action] = \
            np.sum(np.array(self._transition_history[self._current_state][action]), axis=0) / \
            np.sum(np.array(self._transition_history[self._current_state][action]))
        self.__take_step__(arrival_state=arrival_state)
        return

    def value_iteration(self) -> None:
        """
        The value iteration algorithm
        """
        max_Q_change = self._theta  # The biggest absolute Q-value change we experienced in the WHOLE state space
        while max_Q_change >= self._theta:
            max_Q_change = 0
            for state in self._Q.keys():
                for action in self._Q[state].keys():
                    Q_t = self._Q[state][action]  # The current Q-value in (s, a)
                    ####################################################################################################
                    expected_V = 0  # The expected V-function of the arrival state
                    for arrival_state in self._Q.keys():
                        Q_max = self.__Q_max__(state=arrival_state)['value']
                        expected_V += self._transition_function[state][action][arrival_state] * Q_max
                    self._Q[state][action] = self._reward_fun[state][action] + self._gamma * expected_V
                    ####################################################################################################
                    if abs(self._Q[state][action] - Q_t) > max_Q_change:
                        max_Q_change = abs(self._Q[state][action] - Q_t)
        return

    # We overwrite the RL algorithm coming from the parent class
    def reinforcement_learning(self, action: str, arrival_state: int, reward: float) -> None:
        """
        The MF agent will simply use Q-learning to update its Q-values
        :param action: the action taken
        :param arrival_state: the state the agent arrived in after taking the action
        :param reward: the reward received over the state transition
        """
        self.update_model(action=action, arrival_state=arrival_state, reward=reward)
        self.value_iteration()
        super().reinforcement_learning(action=action, arrival_state=arrival_state, reward=reward)
        return
