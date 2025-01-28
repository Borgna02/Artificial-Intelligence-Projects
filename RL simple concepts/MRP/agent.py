from environment import Environment
import numpy as np

class Agent:
    def __init__(self, environment: Environment):

        # Initialize environment
        self.environment = environment

        # Initialize value funcion
        self.value_function = list(environment.rewards)
        
        self.EPSILON = 0.01
        self.GAMMA = 0.7

    def it_bellman(self):
        states = self.environment.states
        rewards = self.environment.rewards
        dynamics = self.environment.dynamics

        # print(f"States {states}, Rewards {rewards}, Dynamics {dynamics}" )

        old_value_function = [0] * len(states)

        iteration = 0
        while any(abs(self.value_function[j] - old_value_function[j]) > self.EPSILON for j in range(len(states))):
            # print(f"Iteration: {iteration}, Value function: {self.value_function}")
            iteration += 1
            
            for i in range(len(states)):
                # Mantengo la vecchia value function
                old_value_function[i] = self.value_function[i]

                # Calcolo quella nuova
                self.value_function[i] = rewards[i] + self.GAMMA *\
                    sum(dynamics[i][j] * old_value_function[j]
                        for j in range(len(states)))

        return self.value_function


    def it_bellman_matrix(self):
        V = np.array(self.value_function).reshape(-1, 1)  # Trasforma in colonna
        R = np.array(self.environment.rewards).reshape(-1, 1)  # Trasforma in colonna
        P = np.array(self.environment.dynamics)  # Matrice di transizione
        
        old_V = np.zeros_like(V)  # Inizializza old_V con zeri
        while np.linalg.norm(V - old_V) > self.EPSILON:  # Norm della differenza tra vettori
            old_V = V.copy()  # Aggiorna old_V
            V = R + self.GAMMA * P @ old_V  # Bellman update con prodotto matriciale
        
        return V.flatten()  # Restituisce un array 1D per compatibilità

    def bellman_matrix(self):
        R = np.array(self.environment.rewards).reshape(-1, 1)  # Trasforma in colonna
        P = np.array(self.environment.dynamics)  # Matrice di transizione
        
        V = np.linalg.inv(np.identity(len(self.environment.rewards)) - self.GAMMA * P) @ R
        
        return V.flatten()