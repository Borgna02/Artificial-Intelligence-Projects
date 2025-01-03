import random
from typing import Literal
from environment import Environment
import numpy as np


class Agent:
    def __init__(self, env: Environment):
        self.env = env
        self.states = set()
        self.actions = set()

        self.N = {}
        self.V = {}

        self.Q = {}

        self.policy = {}

        self.GAMMA = 0.1

    def incremental_mc(self, iterations):
        self.N = {}
        self.V = {}
        for i in range(iterations):
            episode = self.env.get_episode()

            T = len(episode)  # time steps
            for t in range(T):
                state = episode[t][0]

                self.N[state] = self.N.get(state, 0) + 1
                current_V = self.V.get(state, 0)

                # episode[2] is the reward
                G = sum((self.GAMMA ** (t1-t)) *
                        episode[t1][2] for t1 in range(t, T))

                self.V[state] = current_V + 1 / self.N[state] * (G - current_V)

            print(i, " | ", ", ".join(
                f"{key}: {self.V[key]}" for key in sorted(self.V)))

    def incremental_mc_q(self, iterations):
        self.N = {}
        self.Q = {}
        for i in range(iterations):
            episode = self.env.get_episode()
            T = len(episode)

            for t in range(T):
                state = episode[t][0]
                action = episode[t][1]

                self.states.add(state)
                self.actions.add(action)

                self.N[(state, action)] = self.N.get((state, action), 0) + 1
                current_Q = self.Q.get((state, action), 0)

                G = sum((self.GAMMA ** (t1-t)) *
                        episode[t1][2] for t1 in range(t, T))

                self.Q[(state, action)] = current_Q + 1 / \
                    self.N[(state, action)] * (G - current_Q)

            print(i, " | ", ", ".join(
                f"{key}: {self.Q[key]}" for key in self.Q))

    def incremental_mc_q_policy(self):
        self.N = {}
        self.Q = {}
        self.policy = {}
        old_policy = {}

        i = 0
        while not self.policy or any(
            state not in old_policy or
            abs(old_policy[state].get(action, 0) - self.policy[state].get(action, 0)) > 0
            for state in self.policy
            for action in self.policy[state]
        ):  
            episode = self.env.get_episode()
            old_policy = dict(self.policy)
            T = len(episode)

            for t in range(T):
                state = episode[t][0]
                action = episode[t][1]

                self.states.add(state)
                self.actions.add(action)

                self.N[(state, action)] = self.N.get((state, action), 0) + 1
                current_Q = self.Q.get((state, action), 0)

                G = sum((self.GAMMA ** (t1-t)) *
                        episode[t1][2] for t1 in range(t, T))

                self.Q[(state, action)] = current_Q + 1 / \
                    self.N[(state, action)] * (G - current_Q)

            # Aggiorna la policy basata su Q
            for s_i, s in enumerate(self.states, 0):
                # Costruisce Q come dizionario di liste, una lista per ogni stato
                Q = {state: [] for state in self.states}
                # Ordina per azione
                for (state, action), value in sorted(self.Q.items(), key=lambda x: x[0][1].value):
                    if state in Q:
                        # Conserva anche l'azione associata
                        Q[state].append((action, value))

                                    
                # Inizializza la policy dello stato corrente
                self.policy[s] = {action: 0 for action in self.actions} #if not self.policy.get(s) else self.policy[s]

                if Q[s]:  # Solo se ci sono valori per lo stato corrente
                    # Trova l'azione con il valore massimo
                    # max_value = max(Q[s], key=lambda x: x[1])[1]

                    best_action, _ = max(Q[s], key=lambda x: x[1])  # x[1] è il valore associato all'azione
                    self.policy[s][best_action] = 1

                    # for action, value in Q[s]:
                    #     if value == max_value:  # Prima azione con valore massimo
                    #         action_index = list(self.actions).index(
                    #             action)  # Ottieni l'indice dell'azione
                    #         self.policy[s_i][action_index] = 1
                    #         break  # Imposta solo per la prima azione trovata con valore massimo

            print(i, " | ", ", ".join(
                f"{key}: {self.policy[key]}" for key in self.policy))
            i += 1

    def incremental_mc_epsilon_greedy(self):
        self.N = {}
        self.Q = {}
        epsilon = 1
        # if not self.policy:
        #     self.policy = [[-2] * len(self.actions) for _ in range(len(self.states))]
        # old_policy = [[-1] * len(self.actions) for _ in range(len(self.states))]

        self.policy = {}
        old_policy = {}

        i = 0
        while not self.policy or any(
            state not in old_policy or
            abs(old_policy[state].get(action, 0) - self.policy[state].get(action, 0)) > 0
            for state in self.policy
            for action in self.policy[state]
        ):
            episode = self.env.get_episode()
            old_policy = dict(self.policy)

            T = len(episode)
            for t in range(T):
                state = episode[t][0]
                action = episode[t][1]

                self.states.add(state)
                self.actions.add(action)

                self.N[(state, action)] = self.N.get((state, action), 0) + 1
                G = sum(self.GAMMA ** i * episode[i][2] for i in range(t, T))

                current_Q = self.Q.get((state, action), 0)
                self.Q[(state, action)] = current_Q + 1 / \
                    self.N[(state, action)] * (G - current_Q)

            for s in self.states:
                # Costruisce Q come dizionario di liste, una lista per ogni stato
                Q = {state: [] for state in self.states}

                # Ordina per azione
                for (state, action), value in sorted(self.Q.items(), key=lambda x: x[0][1].value):
                    if state in Q:
                        # Conserva anche l'azione associata
                        Q[state].append((action, value))

                # Inizializza la policy dello stato corrente
                self.policy[s] = {action: 0 for action in self.actions} #if not self.policy.get(s) else self.policy[s]

                for action, value in Q[s]:
                    exploration = random.choices([True, False], weights=[
                                                 epsilon, 1 - epsilon])[0]

                    if exploration:
                        choosen_action = random.choices(list(self.actions))[0]
                        self.policy[s][choosen_action] = 1
                    else:
                        # print(Q[s])
                        best_action, _ = max(Q[s], key=lambda x: x[1])  # x[1] è il valore associato all'azione
                        self.policy[s][best_action] = 1
            i += 1
            epsilon = 1/i
