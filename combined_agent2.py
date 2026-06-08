from agents import *
import numpy as np


class BaseCombinedAgent:
    def __init__(self, environment, gamma=0.9, epsilon=0.05, alpha=0.3, theta=0.0001, window_length=10, threshold=0.02):
        self.MF = MFagent(environment=environment, gamma=gamma, epsilon=epsilon, alpha=alpha)
        self.MB = MBagent(environment=environment, gamma=gamma, epsilon=epsilon, theta=theta, window_length=window_length)
        self.threshold = threshold
        self.last_choices = None
        self.value_iteration_calls = 0
        self.value_iteration_history = []
        self.value_iteration_gains = []
        self._episode_value_iteration_calls = 0
        self.decision_control_history = []
        self._episode_decision_control_counts = {'MF': 0, 'MB': 0}

    def action_selection(self):
        mf_choice = self.MF.action_selection()
        mb_choice = self.MB.action_selection()
        self.last_choices = {'MF': mf_choice, 'MB': mb_choice}

        if mf_choice['Q-value'] >= mb_choice['Q-value']:
            self._episode_decision_control_counts['MF'] += 1
            return mf_choice
        self._episode_decision_control_counts['MB'] += 1
        return mb_choice

    def learn(self, action, arrival_state, reward):
        td_error = self.MF.Q_learning(action=action, arrival_state=arrival_state, reward=reward)
        self.MB.update_model(action=action, arrival_state=arrival_state, reward=reward)
        return td_error

    def start_episode(self):
        self._episode_value_iteration_calls = 0
        self._episode_decision_control_counts = {'MF': 0, 'MB': 0}

    def end_episode(self):
        self.value_iteration_history.append(self._episode_value_iteration_calls)
        total_decisions = sum(self._episode_decision_control_counts.values())
        if total_decisions == 0:
            control_rates = {'MF': 0, 'MB': 0}
        else:
            control_rates = {
                controller: count / total_decisions
                for controller, count in self._episode_decision_control_counts.items()
            }
        self.decision_control_history.append(control_rates)

    def value_iteration(self):
        old_Q = {state: actions.copy() for state, actions in self.MB._Q.items()}
        self.value_iteration_calls += 1
        self._episode_value_iteration_calls += 1
        self.MB.value_iteration()
        gain_from_value_iteration = max(abs(self.MB._Q[state][action] - old_Q[state][action]) for state in self.MB._Q.keys() for action in self.MB._Q[state].keys())
        self.value_iteration_gains.append(gain_from_value_iteration)
        return gain_from_value_iteration


class Baseline(BaseCombinedAgent):
    def reinforcement_learning(self, action, arrival_state, reward):
        td_error = self.learn(action, arrival_state, reward)

        if td_error > self.threshold:
            self.value_iteration()



class Adaptive(BaseCombinedAgent):
    def __init__(self, environment, gamma=0.9, epsilon=0.05, alpha=0.3, theta=0.0001, window_length=10, threshold=0.02):
        super().__init__(environment, gamma, epsilon, alpha, theta, window_length, threshold)
        self.visits = {state: {action: 0 for action in self.MF._Q[state].keys()} for state in self.MF._Q.keys()}
        self.stable_planning_useful = {state: {action: True for action in self.MF._Q[state].keys()} for state in self.MF._Q.keys()}

    def reinforcement_learning(self, action, arrival_state, reward):
        old_state = self.MF._current_state
        predicted_state = int(np.argmax(self.MB._transition_function[old_state][action]))
        old_reward_prediction = self.MB._reward_fun[old_state][action]
        old_transition_prediction = np.array(self.MB._transition_function[old_state][action])
        seen_before = self.visits[old_state][action] > 0

        td_error = self.learn(action, arrival_state, reward)
        self.visits[old_state][action] += 1

        reward_prediction_change = abs(self.MB._reward_fun[old_state][action] - old_reward_prediction)
        transition_prediction_change = np.max(np.abs(np.array(self.MB._transition_function[old_state][action]) - old_transition_prediction))
        model_changed = reward_prediction_change > self.threshold or transition_prediction_change > self.threshold
        stable = abs(td_error) < self.threshold
        found_reward = reward > 0
        transition_changed = seen_before and predicted_state != arrival_state
        stable_and_useful = stable and (self.stable_planning_useful[old_state][action] or model_changed)

        if stable_and_useful or found_reward or transition_changed:
            gain_from_value_iteration = self.value_iteration()
            self.stable_planning_useful[old_state][action] = gain_from_value_iteration > self.threshold


class Dissagreement(BaseCombinedAgent):
    def reinforcement_learning(self, action, arrival_state, reward):
        if self.last_choices is None:
            self.action_selection()

        mf_action = self.last_choices['MF']['action']
        mb_action = self.last_choices['MB']['action']

        self.learn(action, arrival_state, reward)

        if mf_action != mb_action:
            self.value_iteration()
