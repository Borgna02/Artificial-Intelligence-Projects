from enum import Enum
import random
from typing import Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Actions(Enum):
    """
    Enum class representing possible actions in the environment.
    Attributes:
        UP (int): Represents the action of moving up.
        LEFT (int): Represents the action of moving left.
        DOWN (int): Represents the action of moving down.
        RIGHT (int): Represents the action of moving right.
        OTHER_SIDE (int): Represents the action of moving to the other side when the agent is on a gate.
    """

    UP = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3
    OTHER_SIDE = 4


class CookingStates(Enum):
    """
    Enum class representing different states in a cooking environment.
    Attributes:
        EB_FOR_SCRAMBLED (int): The agent is searching the egg beater for preparing scrambled eggs.
        EB_FOR_PUDDING (int): The agent is searching the egg beater for preparing pudding.
        PAN (int): The agent is searching for a pan to prepare scrambled eggs.
        OVEN (int): The agent is searching for an oven to prepare pudding.
        COOKING (int): The agent is currently cooking.
    """

    EB_FOR_SCRAMBLED = 0
    EB_FOR_PUDDING = 1
    PAN = 2
    OVEN = 3


class Environment:

    def __init__(self, map_width: int, map_height: int, walls: list[tuple], frying_pans: list[tuple], ovens: list[tuple], egg_beaters: list[tuple], gates: list[tuple], out_of_map_points: list[tuple] = [], save_spreadsheet=False):
        """
        Initializes the environment with the provided parameters.
        Args:
            map_width (int): Width of the map.
            map_height (int): Height of the map.
            walls (list[tuple]): List of tuples ((x1, y1), (x2, y2)) representing walls between two cells.
            frying_pans (list[tuple]): Coordinates of frying pans.
            ovens (list[tuple]): Coordinates of ovens.
            egg_beaters (list[tuple]): Coordinates of egg beaters.
            gates (list[tuple]): Coordinates of gates.
            recipe (Literal): Recipe to follow ("scrambled" or "pudding").
            out_of_map_points (list[tuple]): Optional points considered out of the map.
            spawn_point (tuple): Optional spawn point for the agent.
            save_spreadsheet (bool): Whether to save the dynamics and rewards to Excel.
        """
        self.statistics = dict()

        # Map size
        self.__map_width = map_width
        self.__map_height = map_height

        # Map elements
        self.__walls = walls
        self.__frying_pans = frying_pans
        self.__ovens = ovens
        self.__egg_beaters = egg_beaters
        self.__gates = gates

        # Out of map points
        self._out_of_map_points = out_of_map_points + [(0, y) for y in range(1, self.__map_height+1)] + [(self.__map_width+1, y) for y in range(
            1, self.__map_height+1)] + [(x, 0) for x in range(1, self.__map_width+1)] + [(x, self.__map_height+1) for x in range(1, self.__map_width+1)]

        # Define the actions
        self.actions = [action for action in Actions]

        # Define the states
        self.states = self.__define_states()

        # Define the dynamics and the rewards
        self._dynamics, self._rewards = self.__define_dynamics_and_rewards()

        # Save the dynamics and rewards to Excel
        if save_spreadsheet:
            self.__export_dynamics_to_excel()
            self.__export_rewards_to_excel()

        # Parameters for calculation of the optimal policy
        self.EPSILON = 0.01
        self.GAMMA = 1
        # Inizializza la policy con azioni casuali
        self.policy = {state: {action: 1 / len(self.actions) for action in self.actions}
                       for state in self.states}
        self.value_function_policy = self._rewards_policy()

    def __export_dynamics_to_excel(self):
        """
        Saves dynamics to an Excel file.
        """

        dynamics_data = []
        for state in self.states:
            for action in self.actions:
                for next_state, probability in self._dynamics[state][action].items():
                    dynamics_data.append({
                        'State': state,
                        'Action': action.name,
                        'Next State': next_state,
                        'Probability': probability
                    })

        df_dynamics = pd.DataFrame(dynamics_data)
        df_dynamics.to_excel('dynamics.xlsx', index=False)

    def __export_rewards_to_excel(self):
        """
        Saves rewards to an Excel file.
        """

        rewards_data = []
        for state in self.states:
            for action in self.actions:
                for next_state, reward in self._rewards[state][action].items():
                    rewards_data.append({
                        'State': state,
                        'Action': action.name,
                        'Next State': next_state,
                        'Reward': reward
                    })

        df_rewards = pd.DataFrame(rewards_data)
        df_rewards.to_excel('rewards.xlsx', index=False)

    def __define_dynamics_and_rewards(self):
        """
        Defines the transition dynamics of the environment.
        Returns:
            dict: A nested dictionary representing state-action-next_state transitions.
        """
        dynamics = {state: {action: {next_state: 0 for next_state in self.states}
                            for action in self.actions} for state in self.states}

        rewards = {state: {action: {next_state: -1 for next_state in self.states}
                           for action in self.actions} for state in self.states}

        for current_s in self.states:
            current_x, current_y, current_cooking_state = current_s

            for action in self.actions:

                # Handling the OTHER_SIDE action
                if action == Actions.OTHER_SIDE and (current_x, current_y) in self.__gates:
                    next_x, next_y = self.__get_other_gate(
                        current_x, current_y)
                elif action == Actions.UP:
                    next_x, next_y = current_x, current_y + 1
                elif action == Actions.LEFT:
                    next_x, next_y = current_x - 1, current_y
                elif action == Actions.DOWN:
                    next_x, next_y = current_x, current_y - 1
                elif action == Actions.RIGHT:
                    next_x, next_y = current_x + 1, current_y
                else:
                    next_x, next_y = current_x, current_y

                if (next_x, next_y) in self._out_of_map_points or self.__is_a_wall(current_x, current_y, next_x, next_y):
                    next_x, next_y = current_x, current_y

                next_state = (next_x, next_y, self.__get_next_cooking_state(
                    next_x, next_y, current_cooking_state))

                dynamics[current_s][action][next_state] = 1

                # Se con l'azione si rimane nello stesso stato
                if next_x == current_x and next_y == current_y:
                    rewards[current_s][action][next_state] = -20
                elif current_cooking_state in [CookingStates.EB_FOR_PUDDING, CookingStates.EB_FOR_SCRAMBLED] and (next_x, next_y) in self.__egg_beaters:
                    rewards[current_s][action][next_state] = 200
                elif self.is_goal_state(next_x, next_y, current_cooking_state):
                    rewards[current_s][action][next_state] = 2000

        return dynamics, rewards

    def __define_states(self):
        """
        Defines all possible states in the environment.
        Returns:
            list: A list of tuples representing states in the form (x, y, CookingState).
        """
        cells = [(x, y) for y in range(1, self.__map_height+1)
                 for x in range(1, self.__map_width+1)]

        # Add to the out of map points the points on the border of the map
        self._out_of_map_points = self._out_of_map_points + [(0, y) for y in range(1, self.__map_height+1)] + [(self.__map_width+1, y) for y in range(
            1, self.__map_height+1)] + [(x, 0) for x in range(1, self.__map_width+1)] + [(x, self.__map_height+1) for x in range(1, self.__map_width+1)]

        return [(x, y, cooking_state)
                for x, y in cells for cooking_state in CookingStates]

    def get_episode_start_state(self, recipe: Literal["scrambled", "pudding", "both"], spawn_point: tuple = None):

        # Get the initial random state based on the recipe and the spawn point (if specified)
        initial_state = random.choice([CookingStates.EB_FOR_SCRAMBLED, CookingStates.PAN]) if recipe == "scrambled" else random.choice([CookingStates.EB_FOR_PUDDING, CookingStates.OVEN]) if recipe == "pudding" else random.choice(
            [CookingStates.EB_FOR_SCRAMBLED, CookingStates.PAN, CookingStates.EB_FOR_PUDDING, CookingStates.OVEN])

        if spawn_point:
            valid_states = [state for state in self.states if state[2] ==
                            initial_state and (state[0], state[1]) == spawn_point]
        else:
            valid_states = [state for state in self.states if state[2] == initial_state and (
                state[0], state[1]) not in self._out_of_map_points]

        # Choose a random state
        if len(valid_states) == 0:
            raise ValueError(
                f"No valid states found for recipe '{recipe}'")

        current_state = valid_states[np.random.choice(len(valid_states))]

        # Add the spawn point to the statistics
        if "spawn_points" not in self.statistics:
            self.statistics["spawn_points"] = {}
        self.statistics["spawn_points"][(current_state[0], current_state[1])] = self.statistics["spawn_points"].get(
            (current_state[0], current_state[1]), 0) + 1

        return current_state

    def get_next_state(self, current_state: tuple, action: Actions):
        """
        Gets the next state and reward based on the current state and action.
        Args:
            current_state (tuple): The current state in the form (x, y, CookingState).
            action (Actions): The action to take.
        Returns:
            tuple: A tuple containing the next state and the reward.
        """
        next_states = list(self._dynamics[current_state][action].keys())
        probabilities = list(self._dynamics[current_state][action].values())

        # If there are no transitions defined for the current state and action, return None
        if sum(probabilities) == 0:
            return None, 0

        # Choose a random next state based on the dynamics
        next_state = next_states[np.random.choice(
            len(next_states), p=probabilities)]

        # Get reward
        reward = self._rewards[current_state][action][next_state]

        return next_state, reward

    def draw_map(self, policy=None, cooking_state_for_plc=None, cooking_state_for_rw=None):
        """
        Draws the map and optionally overlays the policy or/and the rewards.
        Args:
            policy (dict): Optional policy to overlay on the map.
            cooking_state_for_rw (CookingStates): Optional state to highlight rewards.
        """
        fig, ax = plt.subplots(
            figsize=(self.__map_width, self.__map_height / 2))

        # Draw the grid
        for x in range(self.__map_width + 2):
            ax.plot([x, x], [1, self.__map_height + 1],
                    color="black", linewidth=0.5)
        for y in range(self.__map_height + 2):
            ax.plot([0, self.__map_width + 1], [y, y],
                    color="black", linewidth=0.5)

        # Draw the walls
        for (x1, y1), (x2, y2) in self.__walls:
            # Vertical wall
            if y1 == y2:
                ax.plot([x1+1, x1+1], [y1, y1+1], color="black", linewidth=3)
            # Horizontal wall
            elif x1 == x2:
                ax.plot([x1, x1+1], [y1+1, y1+1], color="black", linewidth=3)

        # Draw the frying pans with image
        pan_image = plt.imread('img/pan.png')
        for x, y in self.__frying_pans:
            ax.imshow(pan_image, extent=(x, x+1, y, y+1))

        # Draw the ovens with image
        oven_image = plt.imread('img/oven.png')
        for x, y in self.__ovens:
            ax.imshow(oven_image, extent=(x, x+1, y, y+1))

        # Draw the egg beaters
        egg_beater_image = plt.imread('img/beater.png')
        for x, y in self.__egg_beaters:
            ax.imshow(egg_beater_image, extent=(x, x+1, y, y+1))

        # Draw the gates
        for x, y in self.__gates:
            rect = patches.Rectangle(
                (x, y), 1, 1, linewidth=1, edgecolor="yellow", facecolor="yellow", label="Gate")
            ax.add_patch(rect)

        # Draw the policy
        if policy is not None:

            # Add the legend for arrow directions
            direction_labels = {
                Actions.UP: "Up",
                Actions.DOWN: "Down",
                Actions.LEFT: "Left",
                Actions.RIGHT: "Right",
                Actions.OTHER_SIDE: "Other Side"
            }

            for action, label in direction_labels.items():
                if action == Actions.OTHER_SIDE:
                    ax.plot([], [], color="black", label=label, marker=(
                        3, 0, 60), markersize=10, linestyle='None')
                else:
                    ax.plot([], [], color="black", label=label, marker=(
                        3, 0, action.value * 90), markersize=10, linestyle='None')

            arrow_params = {
                "width": 0.01,
                "head_width": 0.1,
                "head_length": 0.1,
                "length_includes_head": True
            }
            action_directions = {
                Actions.UP: (0, 0.5),
                Actions.DOWN: (0, -0.5),
                Actions.LEFT: (-0.5, 0),
                Actions.RIGHT: (0.5, 0),
                Actions.OTHER_SIDE: (0.5, 0.5)
            }
            cooking_state_colors = {
                CookingStates.OVEN: "fuchsia",
                CookingStates.EB_FOR_PUDDING: "orange",
                CookingStates.PAN: "red",
                CookingStates.EB_FOR_SCRAMBLED: "blue",
            }

            if cooking_state_for_plc:
                ax.set_title(f"Policy for Cooking State: {
                             cooking_state_for_plc.name}")

            for (x, y, cooking_state), actions in policy.items():
                if cooking_state_for_plc and cooking_state != cooking_state_for_plc:
                    continue
                # Default to 'black' if the state is not mapped
                color = cooking_state_colors.get(cooking_state, "black")
                for action, active in actions.items():
                    if active == 1:
                        dx, dy = action_directions[action]
                        arrow = patches.FancyArrow(
                            x + 0.5, y + 0.5, dx - 0.07*dx*cooking_state.value, dy - 0.07*dy*cooking_state.value, color=color, **arrow_params
                        )
                        ax.add_patch(arrow)

            # Add the legend for arrow colors
            for cooking_state, color in cooking_state_colors.items():
                ax.plot([], [], color=color, label=cooking_state.name)

        if cooking_state_for_rw is not None:
            # Draw the rewards
            for x, y, current_cooking_state in self.states:
                if current_cooking_state == cooking_state_for_rw:
                    reward = max(self._rewards[current_state][action][(x, y, current_cooking_state)]
                                 for action in self.actions for current_state in self.states if current_state[2] == cooking_state_for_rw)
                    ax.text(x + 0.9, y + 0.1,
                            f"{reward}", ha="right", va="bottom", fontsize=8)

            # Set the plt title as the name of the enum current_state_for_rw
            ax.set_title(f"Rewards per Cooking State: {
                         cooking_state_for_rw.name}")

        # Configure the plot
        ax.set_xlim(1, self.__map_width + 1)
        ax.set_ylim(1, self.__map_height + 1)
        ax.set_aspect('equal')
        ax.axis('on')

        # Add coordinates on axes
        ax.set_xticks([i + 0.5 for i in range(1, self.__map_width + 1)])
        ax.set_yticks([i + 0.5 for i in range(1, self.__map_height + 1)])
        ax.set_xticklabels(range(1, self.__map_width + 1))
        ax.set_yticklabels(range(1, self.__map_height + 1))

        # Add the legend to the right of the map
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(),
                   loc='center left', bbox_to_anchor=(0.72, 0.5))

        plt.show()

    def __get_next_cooking_state(self, next_x, next_y, current_cooking_state):

        # If searching for the egg beater for pudding and reach the egg beater, move to searching for the oven
        if current_cooking_state == CookingStates.EB_FOR_PUDDING and (next_x, next_y) in self.__egg_beaters:
            return CookingStates.OVEN

        # If searching for the egg beater for scrambled and reach the egg beater, move to searching for the frying pan
        elif current_cooking_state == CookingStates.EB_FOR_SCRAMBLED and (next_x, next_y) in self.__egg_beaters:
            return CookingStates.PAN

        else:
            return current_cooking_state

    def __get_other_gate(self, x, y):
        for gate_x, gate_y in self.__gates:
            if gate_x != x and gate_y != y:
                return gate_x, gate_y

        return None

    def __is_a_wall(self, x1, y1, x2, y2):
        return ((x1, y1), (x2, y2)) in self.__walls or ((x2, y2), (x1, y1)) in self.__walls

    def is_goal_state(self, x, y, cooking_state):
        return True if (x, y) in self.__frying_pans and cooking_state == CookingStates.PAN or (x, y) in self.__ovens and cooking_state == CookingStates.OVEN else False

    def _rewards_policy(self):
        # R(s,a)
        # pi(a,s)

        # -> R^pi(s)

        return [sum(self.policy[state][action] * self._rewards[state][action][self.get_next_state(state, action)[0]] for action in self.actions) for state in self.states]

    def _dynamics_policy(self):
        # P(a, s, s') -> P^pi (s, s')

        return [[sum(self.policy[state][action] * self._dynamics[state][action][next_state] for action in self.actions) for state in self.states] for next_state in self.states]

    def policy_iteration_matrix(self):
        """
        Policy Iteration using matrix-based computations and robust Bellman evaluation,
        starting from an already initialized policy in self.policy.
        """
        num_states = len(self.states)
        num_actions = len(self.actions)

        # Convert rewards dictionary to a NumPy array
        rewards_array = np.zeros((num_states, num_actions))
        for s_idx, state in enumerate(self.states):
            for a_idx, action in enumerate(self.actions):
                next_states_rewards = self._rewards[state][action].values()
                rewards_array[s_idx, a_idx] = sum(
                    next_states_rewards) / len(next_states_rewards)

        # Convert dynamics dictionary to a NumPy array
        dynamics_array = np.zeros((num_states, num_actions, num_states))
        for s_idx, state in enumerate(self.states):
            for a_idx, action in enumerate(self.actions):
                for s_next_idx, next_state in enumerate(self.states):
                    dynamics_array[s_idx, a_idx,
                                   s_next_idx] = self._dynamics[state][action][next_state]

        # Convert self.policy to a NumPy array
        policy = np.zeros((num_states, num_actions))
        for s_idx, state in enumerate(self.states):
            for a_idx, action in enumerate(self.actions):
                policy[s_idx, a_idx] = self.policy[state][action]

        max_iterations = 1000
        iteration = 0

        while iteration < max_iterations:
            # Policy Evaluation using Bellman matrix
            def bellman_matrix():
                R_pi = np.sum(policy * rewards_array,
                              axis=1).reshape(-1, 1)  # |S| x 1
                P_pi = np.sum(policy[:, :, None] *
                              dynamics_array, axis=1)  # |S| x |S|

                # Add small regularization to avoid singularity
                epsilon = 1e-10
                matrix = np.eye(num_states) - self.GAMMA * \
                    P_pi + epsilon * np.eye(num_states)

                try:
                    V = np.linalg.inv(matrix) @ R_pi
                except np.linalg.LinAlgError:
                    V = np.linalg.pinv(matrix) @ R_pi

                return V.flatten()

            value_function = bellman_matrix()

            # Policy Improvement
            old_policy = policy.copy()
            Q = rewards_array + self.GAMMA * \
                (dynamics_array @ value_function)  # |S| x |A|
            best_actions = (Q == Q.max(axis=1, keepdims=True)
                            ).astype(float)  # |S| x |A|
            policy = best_actions / \
                best_actions.sum(axis=1, keepdims=True)  # Normalize

            # Check for policy stability
            if np.all(policy == old_policy):
                # print("Policy stable, stopping iteration.")
                break

            iteration += 1

        # if iteration >= max_iterations:
        #     print("Maximum iterations reached.")

        # Convert NumPy policy back to self.policy
        for s_idx, state in enumerate(self.states):
            for a_idx, action in enumerate(self.actions):
                self.policy[state][action] = policy[s_idx, a_idx]

        # Make the policy deterministic by choosing the first action among those with the highest value
        for s_idx, state in enumerate(self.states):
            best_action_idx = np.argmax(policy[s_idx])
            for a_idx, action in enumerate(self.actions):
                self.policy[state][action] = 1.0 if a_idx == best_action_idx else 0.0

        # return self.policy, value_function
