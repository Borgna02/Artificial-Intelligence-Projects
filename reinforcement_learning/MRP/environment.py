import numpy as np

class Environment:
    def __init__(self, n_states):

        self.states = [i for i in range (1, n_states+1)]
        # Genera una matrice 8x8 con valori casuali
        raw_dynamics = np.random.rand(n_states, n_states)
        
        # Normalizza ogni riga in modo che sommi a 1
        normalized_dynamics = (raw_dynamics.T / raw_dynamics.sum(axis=1)).T
        
        # Converti la matrice numpy in una lista di liste Python
        self.dynamics = normalized_dynamics.tolist()
        
        # Aggiungi ricompense casuali per ogni stato
        self.rewards = np.random.randint(0, 10, size=n_states).tolist()
