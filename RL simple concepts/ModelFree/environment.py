import numpy as np


class Environment:
    def __init__(self, n_states, n_actions):

        # Assegno la lista di azioni
        self.actions = [i for i in range(1, n_actions + 1)]

        self.states = [i for i in range(1, n_states + 1)]
        # Genera una matrice 8x8 con valori casuali
        raw_dynamics = np.random.rand(n_actions, n_states, n_states)

        # Normalizza ogni riga lungo la seconda dimensione (asse 1) in modo che sommi a 1
        normalized_dynamics = raw_dynamics / raw_dynamics.sum(axis=2, keepdims=True)

        # Converti la matrice numpy in una lista di liste Python
        
        # P[a][s_t][s_{t+1}]
        self.dynamics = normalized_dynamics.tolist()        
        print(self.dynamics)

        # Aggiungi ricompense casuali per ogni stato
        self.rewards = np.random.randint(0, 10, size=(n_states, n_actions)).tolist()
        
        print(self.rewards)
        

    def get_episode(self):
        episode_length = np.random.randint(4, 8)
        
        # Get starting state randomly        
        current_state = np.random.choice(self.states)
        current_state_index = self.states.index(current_state)
        
        # Initialize episode
        episode = []
        
        for _ in range(episode_length):
            # Get action randomly
            action = np.random.choice(self.actions)
            action_index = self.actions.index(action)
            
            # Get next state according to dynamics
            next_state = np.random.choice(self.states, p=self.dynamics[action_index][current_state_index])
            next_state_index = self.states.index(next_state)
            
            # Get reward
            reward = self.rewards[current_state_index][action_index]
            
            # Update current state
            current_state_index = next_state_index
            
            # Append to episode
            episode.append((int(current_state), int(action), int(reward))) 
            
            current_state = next_state
            current_state_index = next_state_index
            
        return episode