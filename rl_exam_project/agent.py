import random
from typing import Literal
from environment import Environment
from IPython.display import clear_output

class Agent:
    def __init__(self, env: Environment, recipe: Literal["scrambled", "pudding", "both"], spawn_point=None):
        """
        Initialize the agent with the environment, recipe, and optional spawn point.

        Args:
            env (Environment): The environment in which the agent operates.
            recipe (Literal): The specific recipe goal ("scrambled", "pudding", or "both").
            spawn_point: Optional initial spawn point for the agent.
        """

        # Statistics tracking for goal rates and policy changes
        self.statistics = {
            "goal_rate": {
                "successes": {},
                "rate100": {}
            },
            "policy_changes": {
                "exploration": {},
                "exploitation": {}
            }
        }

        # Initialize environment and state-action space
        self.__env = env
        self.__states = env.states
        self.__actions = env.actions

        # Discount factor for future rewards
        self.__GAMMA = 0.7

        # Recipe and spawn point for episode generation
        self.__recipe = recipe
        self.__spawn_point = spawn_point

    def __generate_episode(self, using_policy: bool = False, min_length: int = 1, max_length: int = 100, episode_index: int = 0):
        """
        Generate an episode by simulating transitions in the environment.

        Args:
            using_policy (bool): If True, actions are chosen based on the agent's policy. Otherwise, actions are random.
            min_length (int): Minimum length of the episode.
            max_length (int): Maximum length of the episode.
            episode_index (int): Current episode index, used for statistics.

        Returns:
            list: A list of transitions in the form (state, action, reward).
        """
        # Initialize the starting state
        current_state = self.__env.get_episode_start_state(self.__recipe, self.__spawn_point)
        
        # Clear the output every 200 episodes for better visibility
        if episode_index % 200 == 0:
            clear_output(wait=True)
        print(f"Episode {episode_index} started from {current_state}")

        episode = []
        episode_length = random.randint(min_length, max_length)

        j = 0
        while j < episode_length and not self.__env.is_goal_state(*current_state):
            # Choose an action based on the policy or randomly
            action = random.choices(list(self.policy.get(current_state, self.__actions)), weights=self.policy.get(
                current_state, {a: 1 for a in self.__actions}).values())[0] if using_policy else random.choice(self.__actions)

            # Get the next state and reward from the environment
            next_state, reward = self.__env.get_next_state(current_state, action)

            # Append the transition to the episode
            episode.append((current_state, action, reward))
            current_state = next_state  # Update the current state

            # Check if the current state is a goal state
            success_count = 0
            if self.__env.is_goal_state(*current_state):
                print("Goal reached in episode")
                success_count = 1

            # Update the statistics for successes and success rate over the last 100 episodes
            self.statistics["goal_rate"]["successes"][episode_index] = success_count
            self.statistics["goal_rate"]["rate100"][episode_index] = sum(self.statistics["goal_rate"]["successes"].get(j, 0) for j in range(max(0, episode_index - 100), episode_index)) / 100

            j += 1

        return episode

    def incremental_mc_epsilon_greedy(self):
        """
        Perform incremental Monte Carlo with epsilon-greedy policy improvement.

        Estimates Q-values for state-action pairs and updates the policy incrementally.
        """
        # Initialize counts (N) and Q-values
        self.__N = {state: {action: 0 for action in self.__actions} for state in self.__states}
        self.__Q = {state: {action: 0 for action in self.__actions} for state in self.__states}

        epsilon = 1  # Initial epsilon value

        # Initialize random policy
        self.policy = {state: {action: 0 for action in self.__actions} for state in self.__states}
        for state in self.__states:
            self.policy[state][random.choice(self.__actions)] = 1

        old_policy = {}
        i = 1  # Episode counter

        # Loop until no policy changes
        while (not old_policy or self.statistics["policy_changes"]["exploration"].get(i - 1, 0) + self.statistics["policy_changes"]["exploitation"].get(i - 1, 0) > 0):

            # Generate an episode using the current policy
            episode = self.__generate_episode(using_policy=True, min_length=300, max_length=500, episode_index=i)
            old_policy = dict(self.policy)
            
            # Initialize the set of states in the episode for avoid useless updates in exploitation
            states = set()

            # Process the episode
            for t, (state, action, _) in enumerate(episode):
                states.add(state)
                self.__N[state][action] += 1
                G = sum(self.__GAMMA ** (k - t) * episode[k][2] for k in range(t, len(episode)))
                self.__Q[state][action] += (G - self.__Q[state][action]) / self.__N[state][action]

            # Update the policy
            for s in self.__states:
                old_best_action = max(self.policy[s], key=self.policy[s].get)

                is_exploration = random.random() < epsilon
                if is_exploration:  # Exploration
                    chosen_action = random.choice(self.__actions)
                else:  # Exploitation
                    max_value = max(self.__Q[s].values())
                    best_actions = [a for a, q in self.__Q[s].items() if q == max_value]
                    chosen_action = random.choice(best_actions) if s in states else best_actions[0]

                if chosen_action != old_best_action:
                    change_type = "exploration" if is_exploration else "exploitation"
                    self.statistics["policy_changes"][change_type][i] = self.statistics["policy_changes"][change_type].get(i, 0) + 1

                self.policy[s] = {a: 1 if a == chosen_action else 0 for a in self.__actions}

            # Log changes and update epsilon
            print(f"Policy changes for episode {i} (eps: {epsilon}): exploration: {self.statistics['policy_changes']['exploration'].get(
                i, 0)}, exploitation: {self.statistics['policy_changes']['exploitation'].get(i, 0)}")
            
            i += 1
            epsilon = 1 / (i ** 0.25)  # Decay epsilon

        print(f"Epsilon after {i} episodes: {epsilon}")
