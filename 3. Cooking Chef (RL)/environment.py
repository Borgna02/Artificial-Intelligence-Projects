from enum import Enum
import random
from typing import Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BEATER_URL = 'img/beater.png'
OVEN_URL = 'img/oven.png'
PAN_URL = 'img/pan.png'
GATE_URL = 'img/gate.png'


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
        Initialize the environment with the provided parameters.

        Args:
            map_width (int): Width of the map.
            map_height (int): Height of the map.
            walls (list[tuple]): List of tuples ((x1, y1), (x2, y2)) representing walls between two cells.
            frying_pans (list[tuple]): Coordinates of frying pans.
            ovens (list[tuple]): Coordinates of ovens.
            egg_beaters (list[tuple]): Coordinates of egg beaters.
            gates (list[tuple]): Coordinates of gates.
            out_of_map_points (list[tuple]): Optional points considered out of the map.
            save_spreadsheet (bool): Whether to save the dynamics and rewards to Excel.
        """
        self.statistics = {"spawn_points": {}}

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
        self._out_of_map_points = [
            *out_of_map_points,
            *[(0, y) for y in range(1, self.__map_height + 1)],
            *[(self.__map_width + 1, y) for y in range(1, self.__map_height + 1)],
            *[(x, 0) for x in range(1, self.__map_width + 1)],
            *[(x, self.__map_height + 1) for x in range(1, self.__map_width + 1)]
        ]

        # Define the actions
        self.actions = [action for action in Actions]

        # Define the states
        self.states = self.__define_states()

        # Define the dynamics and the rewards
        self.__dynamics, self.__rewards = self.__define_dynamics_and_rewards()

        # Initialize random policy
        self.policy = {
            state: {action: 1 / len(self.actions) for action in self.actions}
            for state in self.states
        }

        # Discount factor for future rewards
        self.__GAMMA = 0.7

        # Save the dynamics and rewards to Excel
        if save_spreadsheet:
            self.__export_dynamics_to_excel()
            self.__export_rewards_to_excel()

    def __define_states(self):
        """
        Defines all possible states in the environment.
        Returns:
            list: A list of tuples representing states in the form (x, y, CookingState).
        """
        cells = [(x, y) for y in range(1, self.__map_height+1)
                 for x in range(1, self.__map_width+1)]

        return [(x, y, cooking_state)
                for x, y in cells for cooking_state in CookingStates]

    def __define_dynamics_and_rewards(self):
        """
        Defines the transition dynamics of the environment.
        Returns:
            dict: A nested dictionary representing state-action-next_state transitions.
        """
        dynamics = {state: {action: {next_state: 0 for next_state in self.states} for action in self.actions} for state in self.states}

        rewards = {state: {action: {next_state: -1 for next_state in self.states} for action in self.actions} for state in self.states}

        for current_s in self.states:
            current_x, current_y, current_cooking_state = current_s

            for action in self.actions:

                # Handling the OTHER_SIDE action
                if action == Actions.OTHER_SIDE and (current_x, current_y) in self.__gates:
                    next_x, next_y = self.__get_other_gate(
                        current_x, current_y)
                # Handling the rest of the actions
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

                # Check if the next state is out of the map or a wall
                if (next_x, next_y) in self._out_of_map_points or self.__is_a_wall(current_x, current_y, next_x, next_y):
                    next_x, next_y = current_x, current_y

                # Get the next cooking state
                next_state = (next_x, next_y, self.__get_next_cooking_state(
                    next_x, next_y, current_cooking_state))

                # Update the dynamics
                dynamics[current_s][action][next_state] = 1

                # Update the rewards
                if next_x == current_x and next_y == current_y:
                    rewards[current_s][action][next_state] = -20  # Penalize hitting a wall or teleporting being not on a gate
                elif current_cooking_state in [CookingStates.EB_FOR_PUDDING, CookingStates.EB_FOR_SCRAMBLED] and (next_x, next_y) in self.__egg_beaters:
                    rewards[current_s][action][next_state] = 200  # Reward for finding the egg beater
                elif self.is_goal_state(next_x, next_y, current_cooking_state):
                    rewards[current_s][action][next_state] = 2000  # Reward for reaching the goal

        return dynamics, rewards

    def __export_dynamics_to_excel(self):
        """
        Saves dynamics to an Excel file.
        """

        dynamics_data = []
        for state in self.states:
            for action in self.actions:
                for next_state, probability in self.__dynamics[state][action].items():
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
                for next_state, reward in self.__rewards[state][action].items():
                    rewards_data.append({
                        'State': state,
                        'Action': action.name,
                        'Next State': next_state,
                        'Reward': reward
                    })

        df_rewards = pd.DataFrame(rewards_data)
        df_rewards.to_excel('rewards.xlsx', index=False)

    def get_episode_start_state(self, recipe: Literal["scrambled", "pudding", "both"], spawn_point: tuple = None):
        """
        Get the starting state for an episode based on the recipe and spawn point.

        Args:
            recipe (Literal): Recipe goal ("scrambled", "pudding", or "both").
            spawn_point (tuple, optional): Specific spawn point coordinates.

        Returns:
            tuple: Initial state for the episode.
        """
        # Determine initial state based on the recipe
        recipe_states = {
            "scrambled": [CookingStates.EB_FOR_SCRAMBLED, CookingStates.PAN],
            "pudding": [CookingStates.EB_FOR_PUDDING, CookingStates.OVEN],
            "both": [
                CookingStates.EB_FOR_SCRAMBLED, CookingStates.PAN,
                CookingStates.EB_FOR_PUDDING, CookingStates.OVEN
            ]
        }
        initial_state = random.choice(recipe_states[recipe])

        # Filter valid states
        valid_states = [
            state for state in self.states
            if state[2] == initial_state and (
                (state[0], state[1]) == spawn_point if spawn_point else (state[0], state[1]) not in self._out_of_map_points
            )
        ]

        # Choose a random valid state or fallback to a default
        current_state = random.choice(valid_states) if valid_states else self.states[0]
        spawn_point_key = (current_state[0], current_state[1])

        # Update spawn point statistics
        self.statistics["spawn_points"][spawn_point_key] = self.statistics["spawn_points"].get(spawn_point_key, 0) + 1

        return current_state

    def get_next_state(self, current_state: tuple, action: Actions):
        """
        Get the next state and reward based on the current state and action.

        Args:
            current_state (tuple): The current state in the form (x, y, CookingState).
            action (Actions): The action to take.

        Returns:
            tuple: The next state and its associated reward.
        """
        # Retrieve possible next states and their probabilities
        next_states = list(self.__dynamics[current_state][action].keys())
        probabilities = list(self.__dynamics[current_state][action].values())

        # Select the next state based on probabilities or default to current state
        next_state = (
            next_states[np.random.choice(len(next_states), p=probabilities)]
            if sum(probabilities) > 0 else current_state
        )

        # Retrieve the reward for the transition
        reward = self.__rewards[current_state][action].get(next_state, 0)

        return next_state, reward

    def draw_map(self, policy=None, cooking_state_for_plc=None, cooking_state_for_rw=None):
        """
        Draw the map and optionally overlay the policy and/or rewards.

        Args:
            policy (dict): Optional policy to overlay on the map.
            cooking_state_for_plc (CookingStates): Optional state to highlight policy.
            cooking_state_for_rw (CookingStates): Optional state to highlight rewards.
        """
        fig, ax = plt.subplots(figsize=(self.__map_width, self.__map_height / 2))

        # Draw grid
        for x in range(self.__map_width + 2):
            ax.plot([x, x], [1, self.__map_height + 1], color="black", linewidth=0.5)
        for y in range(self.__map_height + 2):
            ax.plot([0, self.__map_width + 1], [y, y], color="black", linewidth=0.5)

        # Draw walls
        for (x1, y1), (_, y2) in self.__walls:
            ax.plot([x1 + 1, x1 + 1], [y1, y1 + 1], color="black", linewidth=3) if y1 == y2 else ax.plot([x1, x1 + 1], [y1 + 1, y1 + 1], color="black", linewidth=3)

        # Draw elements with images
        for img_url, locations in {BEATER_URL: self.__egg_beaters, OVEN_URL: self.__ovens, PAN_URL: self.__frying_pans, GATE_URL: self.__gates}.items():
            img = plt.imread(img_url)
            for x, y in locations:
                ax.imshow(img, extent=(x, x + 1, y, y + 1))

        # Draw policy arrows
        if policy:
            arrow_params = {"width": 0.01, "head_width": 0.1, "head_length": 0.1, "length_includes_head": True}
            action_directions = {
                Actions.UP: (0, 0.5), Actions.DOWN: (0, -0.5), Actions.LEFT: (-0.5, 0), Actions.RIGHT: (0.5, 0), Actions.OTHER_SIDE: (0.5, 0.5)
            }

            cooking_state_colors = {
                CookingStates.OVEN: "fuchsia", CookingStates.EB_FOR_PUDDING: "green",
                CookingStates.PAN: "red", CookingStates.EB_FOR_SCRAMBLED: "blue"
            }

            if cooking_state_for_plc:
                ax.set_title(f"Policy for Cooking State: {cooking_state_for_plc.name}")

            for (x, y, cooking_state), actions in policy.items():
                if cooking_state_for_plc and cooking_state != cooking_state_for_plc:
                    continue
                color = cooking_state_colors.get(cooking_state, "black")
                for action, active in actions.items():
                    if active:
                        dx, dy = action_directions[action]
                        ax.add_patch(patches.FancyArrow(x + 0.5, y + 0.5, dx, dy, color=color, **arrow_params))

            for cooking_state, color in cooking_state_colors.items():
                ax.plot([], [], color=color, label=cooking_state.name)

            for action in Actions:
                angle = action.value * 90 if action != Actions.OTHER_SIDE else 60
                ax.plot([], [], color="black", label=action.name, marker=(3, 0, angle), markersize=10, linestyle='None')

        # Draw rewards
        if cooking_state_for_rw:
            for x, y, current_cooking_state in self.states:
                if current_cooking_state == cooking_state_for_rw:
                    reward = max(
                        self.__rewards[current_state][action].get((x, y, self.get_next_state(current_state, action)[0][2]), 0)
                        for action in self.actions for current_state in self.states if current_state[2] == cooking_state_for_rw
                    )
                    ax.text(x + 0.9, y + 0.1, f"{reward}", ha="right", va="bottom", fontsize=8)
            ax.set_title(f"Rewards for Cooking State: {cooking_state_for_rw.name}")

        # Configure plot
        ax.set_xlim(1, self.__map_width + 1)
        ax.set_ylim(1, self.__map_height + 1)
        ax.set_aspect('equal')
        ax.set_xticks([i + 0.5 for i in range(1, self.__map_width + 1)])
        ax.set_yticks([i + 0.5 for i in range(1, self.__map_height + 1)])
        ax.set_xticklabels(range(1, self.__map_width + 1))
        ax.set_yticklabels(range(1, self.__map_height + 1))

        # Add legend if there are any elements
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), loc='center left', bbox_to_anchor=(0.72, 0.5))

        plt.show()

    def __get_next_cooking_state(self, next_x, next_y, current_cooking_state):
        """
        Determine the next cooking state based on the current state and position.

        Args:
            next_x (int): X-coordinate of the next position.
            next_y (int): Y-coordinate of the next position.
            current_cooking_state (CookingStates): Current cooking state.

        Returns:
            CookingStates: The next cooking state.
        """
        if current_cooking_state == CookingStates.EB_FOR_PUDDING and (next_x, next_y) in self.__egg_beaters:
            return CookingStates.OVEN
        if current_cooking_state == CookingStates.EB_FOR_SCRAMBLED and (next_x, next_y) in self.__egg_beaters:
            return CookingStates.PAN
        return current_cooking_state

    def __get_other_gate(self, x, y):
        """
        Get the coordinates of the other gate, if available.

        Args:
            x (int): X-coordinate of the current gate.
            y (int): Y-coordinate of the current gate.

        Returns:
            tuple or None: Coordinates of the other gate or None if not found.
        """
        return next(((gate_x, gate_y) for gate_x, gate_y in self.__gates if (gate_x, gate_y) != (x, y)), None)

    def __is_a_wall(self, x1, y1, x2, y2):
        """
        Check if a wall exists between two points.

        Args:
            x1 (int): X-coordinate of the first point.
            y1 (int): Y-coordinate of the first point.
            x2 (int): X-coordinate of the second point.
            y2 (int): Y-coordinate of the second point.

        Returns:
            bool: True if a wall exists, False otherwise.
        """
        return ((x1, y1), (x2, y2)) in self.__walls or ((x2, y2), (x1, y1)) in self.__walls

    def is_goal_state(self, x, y, cooking_state):
        """
        Check if the current state is a goal state.

        Args:
            x (int): X-coordinate of the position.
            y (int): Y-coordinate of the position.
            cooking_state (CookingStates): Current cooking state.

        Returns:
            bool: True if the state is a goal state, False otherwise.
        """
        return (x, y) in self.__frying_pans and cooking_state == CookingStates.PAN or \
            (x, y) in self.__ovens and cooking_state == CookingStates.OVEN

    def policy_iteration_matrix(self):
        """
        Policy Iteration using matrix-based computations, starting from an initialized policy in self.policy.
        """
        num_states = len(self.states)
        num_actions = len(self.actions)

        # Convert rewards to a NumPy array
        rewards_array = np.array([
            [sum(self.__rewards[state][action].values()) / len(self.__rewards[state][action]) for action in self.actions]
            for state in self.states
        ])

        # Convert dynamics to a NumPy array
        dynamics_array = np.array([
            [[self.__dynamics[state][action][next_state] for next_state in self.states] for action in self.actions]
            for state in self.states
        ])

        # Convert policy to a NumPy array
        policy = np.array([
            [self.policy[state][action] for action in self.actions] for state in self.states
        ])

        def bellman_matrix():
            """
            Compute the value function using the Bellman equation.
            """
            R_pi = np.sum(policy * rewards_array, axis=1).reshape(-1, 1)  # |S| x 1
            P_pi = np.sum(policy[:, :, None] * dynamics_array, axis=1)  # |S| x |S|
            matrix = np.eye(num_states) - self.__GAMMA * P_pi + 1e-10 * np.eye(num_states)  # Regularization

            try:
                V = np.linalg.inv(matrix) @ R_pi
            except np.linalg.LinAlgError:
                V = np.linalg.pinv(matrix) @ R_pi

            return V.flatten()

        for _ in range(1000):  # Maximum iterations
            value_function = bellman_matrix()

            # Policy Improvement
            Q = rewards_array + self.__GAMMA * (dynamics_array @ value_function)  # |S| x |A|
            best_actions = (Q == Q.max(axis=1, keepdims=True)).astype(float)  # |S| x |A|
            new_policy = best_actions / best_actions.sum(axis=1, keepdims=True)  # Normalize

            if np.all(policy == new_policy):  # Check for policy stability
                break

            policy = new_policy

        # Update self.policy to reflect the deterministic policy
        for s_idx, state in enumerate(self.states):
            best_action_idx = np.argmax(policy[s_idx])
            for a_idx, action in enumerate(self.actions):
                self.policy[state][action] = 1.0 if a_idx == best_action_idx else 0.0
