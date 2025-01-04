from collections import defaultdict
import random
import time
from typing import Literal
from environment import Environment, Actions, CookingStates
import numpy as np
import pandas as pd


class Agent:
    def __init__(self, env: Environment, recipe: Literal["scrambled", "pudding", "both"], spawn_point=None):
        self.statistics = dict()

        self.env = env
        self.states = env.states
        self.actions = env.actions

        self.N = {}
        self.V = {}
        self.Q = {}

        self.policy = {}

        self.GAMMA = 0.7

        self.__recipe = recipe
        self.__spawn_point = spawn_point

    def __generate_episode(self, using_policy: bool = False, min_length: int = 1, max_length: int = 100, i=0):
        current_state = self.env.get_episode_start_state(
            self.__recipe, self.__spawn_point)
        print(f"\n\nEpisode {i} started from {current_state}")
        episode = []
        for _ in range(random.randint(min_length, max_length)):

            if not using_policy:
                action = random.choice(self.actions)
            else:
                try:
                    action = random.choices(
                        list(self.policy[current_state]), weights=self.policy[current_state].values())[0]
                except KeyError:
                    action = random.choice(self.actions)

            next_state, reward = self.env.get_next_state(current_state, action)

            episode.append((current_state, action, reward))

            # print(f"State: {current_state}, Action: {action}, Reward: {reward}, Next State: {next_state}")
            current_state = next_state

            if "goal_rate" not in self.statistics:
                self.statistics["goal_rate"] = {"n": 0, "successes": {}}
            if self.env.is_goal_state(*current_state):
                print("Goal")

                self.statistics["goal_rate"]["n"] = self.statistics["goal_rate"].get(
                    "n", 0) + 1
                self.statistics["goal_rate"]["successes"][i] = self.statistics["goal_rate"]["successes"].get(
                    i-1, 0) + 1
                break
            else:
                self.statistics["goal_rate"]["n"] = self.statistics["goal_rate"].get(
                    "n", 0) + 1
                self.statistics["goal_rate"]["successes"][i] = self.statistics["goal_rate"]["successes"].get(
                    i-1, 0) + 0

        return episode

    def incremental_mc_epsilon_greedy(self):
        self.statistics["policy_changes"] = {"exploration": {}, "exploitation": {}}

        # Inizializza i contatori N e Q per tutte le azioni in ogni stato
        self.N = {state: {action: 0 for action in self.actions}
                  for state in self.states}
        self.Q = {state: {action: 0 for action in self.actions}
                  for state in self.states}

        epsilon = 0.1  # Valore iniziale di epsilon per la strategia epsilon-greedy

        # Inizializza la policy con azioni casuali per ogni stato
        self.policy = {state: {action: 0 for action in self.actions}
                       for state in self.states}
        for state in self.states:
            chosen_action = random.choice(self.actions)
            self.policy[state][chosen_action] = 1

        old_policy = {}
        i = 1  # Contatore per gli episodi

        while (not old_policy or self.statistics["policy_changes"]["exploration"].get(i-1, 0) + self.statistics["policy_changes"]["exploitation"].get(i-1, 0) > 0):

            # Genera un episodio utilizzando la policy corrente
            episode = self.__generate_episode(
                using_policy=True, min_length=300, max_length=500, i=i)
            old_policy = dict(self.policy)  # Salva la vecchia policy

            states = set()
            T = len(episode)  # Lunghezza dell'episodio
            for t in range(T):
                state, action, reward = episode[t]
                states.add(state)
                # Aggiorna il conteggio N per la coppia stato-azione
                self.N[state][action] += 1

                # Calcola il ritorno (G) dall'attuale passo fino alla fine dell'episodio
                G = sum(self.GAMMA ** (k-t) *
                        episode[k][2] for k in range(t, T))

                # Aggiorna Q usando il metodo incrementale Monte Carlo
                current_Q = self.Q[state][action]
                self.Q[state][action] = current_Q + \
                    (1 / self.N[state][action]) * (G - current_Q)

            # Convert Q to a DataFrame and save to a csv file
            # q_data = []
            # for state in self.Q:
            #     for action in self.Q[state]:
            #         q_data.append([state, action, self.Q[state][action]])

            # df = pd.DataFrame(q_data, columns=['State', 'Action', 'Q-value'])
            # df.to_csv('Q_values.csv', index=False)
            # time.sleep(2)

            # Aggiorna la policy epsilon-greedy per ogni stato incontrato nell'episode
            for s in self.states:
                
                old_best_action = [action for action, value in self.policy[s].items() if value == 1][0]
                
                
                # # Resetta la policy dello stato corrente
                # self.policy[s] = {action: 0 for action in self.actions}

                # Decidi se esplorare o sfruttare
                if random.random() < epsilon:  # Esplorazione
                    chosen_action = random.choice(self.actions)
                    self.policy[s][old_best_action] = 0
                    self.policy[s][chosen_action] = 1
                    if old_best_action != chosen_action:
                        self.statistics["policy_changes"]["exploration"][i] = self.statistics["policy_changes"]["exploration"].get(
                            i, 0) + 1
                else:  # Sfruttamento
                    max_value = max(self.Q[s].values())
                    best_actions = [
                        action for action, value in self.Q[s].items() if value == max_value]
                    chosen_action = random.choice(
                        best_actions) if s in states else best_actions[0]
                    self.policy[s][old_best_action] = 0
                    self.policy[s][chosen_action] = 1

                    if old_best_action != chosen_action:
                        self.statistics["policy_changes"]["exploitation"][i] = self.statistics["policy_changes"]["exploitation"].get(
                            i, 0) + 1
                        
            print(f"Policy changes for episode {i} (eps: {epsilon}): exploration: {self.statistics['policy_changes']['exploration'].get(i, 0)}, exploitation: {self.statistics['policy_changes']["exploitation"].get(i, 0)}")

            # Incrementa il contatore degli episodi e aggiorna epsilon
            i += 1
            # epsilon = 1 / (i ** 0.25)  # Decadimento perfetto
            epsilon = 1 / (i ** 0.25)  # Decadimento perfetto

        print(f"Epsilon dopo il {i}° episodio: {epsilon}")


