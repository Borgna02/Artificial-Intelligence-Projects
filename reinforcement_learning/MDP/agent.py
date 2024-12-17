from typing import Literal
from environment import Environment
import numpy as np


class Agent:
    def __init__(self, environment: Environment):

        # Initialize environment
        self.env = environment


        # Initialize random policy
        raw_policy = np.random.rand(
            len(environment.actions), len(environment.states))
        normalized_policy = raw_policy / raw_policy.sum(axis=0, keepdims=True)
        self.policy = normalized_policy.tolist()
        print(np.array(self.policy))

        # Initialize value funcion [a][s]
        self.value_function_policy = self._rewards_policy()
        
        self.EPSILON = 0.01
        self.GAMMA = 0.7

    def bellman_backup(self, to_print=True, val:Literal["old", "current"]="old"):
        # Inizializza la funzione di valore precedente
        old_value_function = [0] * len(self.env.states)

        # Continua ad aggiornare finché la differenza tra le due funzioni di valore non è minore di EPSILON
        while any(
            abs(old_value_function[i] - self.value_function_policy[i]) > self.EPSILON
            for i in range(len(self.value_function_policy))
        ):
            if to_print:
                print(self.value_function_policy)
            # Copia la vecchia funzione di valore
            old_value_function = list(self.value_function_policy)

            # Aggiorna la funzione di valore basata sulla formula di Bellman
            self.value_function_policy = [
                self._rewards_policy()[i_state] + self.GAMMA * sum(
                    self._dynamics_policy()[i_state][i_state1] * (old_value_function[i_state1] if val == "old" else self.value_function_policy)
                    for i_state1 in range(len(self.env.states))
                )
                for i_state in range(len(self.env.states))
            ]
            
    def _rewards_policy(self):
        # R(s,a)
        # pi(a,s)
        
        # -> R^pi(s)

        return [sum(self.policy[i_action][i_state] * self.env.rewards[i_state][i_action] for i_action in range(len(self.env.actions))) for i_state in range(len(self.env.states))]

    def _dynamics_policy(self):
        # P(a, s, s') -> P^pi (s, s')
        
        return [[sum(self.policy[i_action][i_state] * self.env.dynamics[i_action][i_state][i_state1] for i_action in range(len(self.env.actions))) for i_state in range(len(self.env.states))] for i_state1 in range(len(self.env.states))]
    
    
    def policy_iteration(self):
        old_policy = [[-1] * len(self.policy[0])] * len(self.policy)
        while np.any(np.greater(np.array(self.policy) - np.array(old_policy), 0)):
            # Policy evaluation
            self.bellman_backup(False)
            # Policy improvement
            old_policy = np.array(self.policy).tolist()
            for s in range(len(self.env.states)):
                q_policy = [0] * len(self.env.actions)
                for a in range(len(self.env.actions)):
                    q_policy[a] = self.env.rewards[s][a] + self.GAMMA * sum(self.env.dynamics[a][s][s1] * self.value_function_policy[s1] for s1 in range(len(self.env.states)))
                    self.policy[a][s] = 0
                
                for a in range(len(self.env.actions)):
                    if q_policy[a] == max(q_policy):
                        self.policy[a][s] = 1 
                        break  
                    
        return self.policy
    
    def bellman_matrix(self):
        R = np.array(self._rewards_policy()).reshape(-1, 1)  # Trasforma in colonna
        P = np.array(self._dynamics_policy())  # Matrice di transizione
        
        V = np.linalg.inv(np.identity(len(self.env.states)) - self.GAMMA * P) @ R
        
        return V.flatten()