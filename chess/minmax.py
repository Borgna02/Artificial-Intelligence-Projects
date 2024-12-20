

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import heapq
import random
import time
from typing import Callable
from enum import Enum

import joblib


class Algorithms(Enum):
    MIN_MAX = "min_max"
    FAIL_HARD_ALPHA_BETA = "fail_hard_alpha_beta"
    FAIL_SOFT_ALPHA_BETA = "fail_soft_alpha_beta"
    BRANCHING_LIMIT = "branching_limit"
    PRED_BLMINMAX = "pred_blminmax"
    MULTI_INPUT_PRED_BLMINMAX = "multi_input_pred_blminmax"


class MinMax:
    def __init__(self, get_children: Callable, evaluate: Callable, engine: Algorithms):
        self.maximizing_player = None
        self.get_children = get_children
        self.H_0 = evaluate
        self.model = None
        self.starting_L = None

        self.hits = 0
        self.cache_hits = 0

        match engine:
            case Algorithms.MIN_MAX:
                self.engine = self.minmax
            case Algorithms.FAIL_HARD_ALPHA_BETA:
                self.engine = self.fhabminmax
            case Algorithms.FAIL_SOFT_ALPHA_BETA:
                self.engine = self.fsabminmax
            case Algorithms.BRANCHING_LIMIT:
                self.engine = self.blminmax
            case Algorithms.PRED_BLMINMAX:
                self.engine = self.pred_blminmax
            case Algorithms.MULTI_INPUT_PRED_BLMINMAX:
                self.engine = self.mi_pred_blminmax
            case _:
                raise ValueError(
                    F"Invalid engine type. Choose between {', '.join([engine.value for engine in Algorithms])}")

    def minmax(self, state, L, maximizing_player=True):
        # Update the current player
        self.maximizing_player = maximizing_player

        # Store the children to avoid redundant calculations
        children = self.get_children(state)

        # Terminal condition: depth is zero or no children remain
        if L == 0 or not children:
            return self.H_0(state), state

        # Initialize the best value and the best node
        best_value = float('-inf') if maximizing_player else float('inf')
        best_children = []

        # Iterate through the children
        for child in children:
            child_value, _ = self.minmax(child, L - 1, not maximizing_player)

            if maximizing_player:
                if child_value > best_value:
                    best_value = child_value
                    best_children = [child]
                elif child_value == best_value:
                    best_children.append(child)
            else:
                if child_value < best_value:
                    best_value = child_value
                    best_children = [child]
                elif child_value == best_value:
                    best_children.append(child)

        # Choose a random child among those with the best value
        best_child = random.choice(best_children) if best_children else None

        return best_value, best_child


    def fhabminmax(self, state, L, alpha=float('-inf'), beta=float('inf'), maximizing_player=True):

        # Aggiorna il giocatore corrente
        self.maximizing_player = maximizing_player

        # Memorizza i figli per evitare calcoli ripetuti
        children = self.get_children(state)

        # Condizione terminale: profondità zero o stato terminale
        if L == 0 or not children:
            return self.H_0(state), state

        # Inizializza il valore migliore e il nodo migliore
        best_value = float('-inf') if maximizing_player else float('inf')
        best_children = []

        # Ciclo sui figli
        for child in children:
            child_value, _ = self.fhabminmax(
                child, L - 1, alpha, beta, not maximizing_player)

            if maximizing_player:
                if child_value > best_value:
                    best_value = child_value
                    best_children = [child]
                elif child_value == best_value:
                    best_children.append(child)
                alpha = max(alpha, best_value)
            else:
                if child_value < best_value:
                    best_value = child_value
                    best_children = [child]
                elif child_value == best_value:
                    best_children.append(child)
                beta = min(beta, best_value)
            if beta < alpha:
                break  # Potatura Beta

        # Scegli un figlio casuale tra quelli con il valore migliore
        best_child = random.choice(best_children) if best_children else None

        return best_value, best_child

    def fsabminmax(self, state, L, alpha=float('-inf'), beta=float('inf'), maximizing_player=True):
        # Update the current player
        self.maximizing_player = maximizing_player

        children = self.get_children(state)

        # Terminal condition: depth is zero or no children remain
        if L == 0 or not children:
            return self.H_0(state), state

        # Initialize the best value and the best nodes
        best_value = float('-inf') if maximizing_player else float('inf')
        best_children = []

        # Iterate through the children
        for child in children:
            child_value, _ = self.fsabminmax(
                child, L - 1, alpha, beta, not maximizing_player)

            if maximizing_player:
                if child_value > best_value:
                    best_value = child_value
                    # Reset the list of best children
                    best_children = [child]
                elif child_value == best_value:
                    # Add the child with the same value
                    best_children.append(child)
                alpha = max(alpha, best_value)
            else:
                if child_value < best_value:
                    best_value = child_value
                    # Reset the list of best children
                    best_children = [child]
                elif child_value == best_value:
                    # Add the child with the same value
                    best_children.append(child)
                beta = min(beta, best_value)

            if beta <= alpha:
                break  # Beta pruning

        # Choose a random child among those with the best value
        best_child = random.choice(best_children) if best_children else None

        return best_value, best_child


    def blminmax(self, state, L, l=0, BF=10, alpha=float('-inf'), beta=float('inf'), maximizing_player=True):
        """
            MinMax with Alpha-Beta pruning and branching factor limit, with child ordering based on lookahead.

            Args:
                state: Current game state.
                L: Maximum depth of the search tree.
                l: Lookahead depth for ordering children.
                BF: Limit on the number of children to explore.
                alpha: Alpha value for pruning.
                beta: Beta value for pruning.
                maximizing_player: True if it's the maximizing player's turn.

            Returns:
                tuple: (best value, best child state)
        """
        self.maximizing_player = maximizing_player
        
        # Transposition table for avoiding redundant calculations
        if self.starting_L is None:
            self.starting_L = L
            self.transposition_table = {}

        
        try:
            state_key = (hash(state), L, l, BF, maximizing_player)
        except TypeError:
            state_key = (hash(str(state)), L, l, BF, maximizing_player) # Fallback for unhashable states
        
        self.hits += 1
        if state_key in self.transposition_table:
            self.cache_hits += 1   
            return self.transposition_table[state_key]
        
        
        children = self.get_children(state)

        # Terminal condition
        if L == 0 or not children:
            return self.H_0(state), state

        best_value = float('-inf') if maximizing_player else float('inf')
        best_children = []

        # Determine the effective lookahead depth
        effective_l = min(l, L-1)

        # Evaluate children in parallel
        def evaluate_child(child):
            if effective_l > 0:
                return child, self.blminmax(child, effective_l, l=0, BF=BF, alpha=alpha, beta=beta, maximizing_player=not maximizing_player)[0]
            else:
                return child, self.H_0(child)

        with ThreadPoolExecutor() as executor:
            evaluated_children = list(executor.map(evaluate_child, children))

        # Sort and limit children
        if len(children) > BF:
            if effective_l == 0:
                limited_children = heapq.nlargest(BF, evaluated_children, key=lambda x: x[1] if maximizing_player else -x[1])   
            else:
                limited_children = heapq.nlargest(BF, evaluated_children, key=lambda x: x[1] if (not maximizing_player if effective_l % 2 == 0 else maximizing_player) else -x[1])
        else:
            limited_children = evaluated_children

   
        
        # Iterate over the limited children
        for child, _ in limited_children:
            child_value, _ = self.blminmax(child, L - 1, l, BF, alpha, beta, not maximizing_player)

            if maximizing_player:
                if child_value > best_value:
                    best_value = child_value
                    best_children = [child]
                elif child_value == best_value:
                    best_children.append(child)
                alpha = max(alpha, child_value)
            else:
                if child_value < best_value:
                    best_value = child_value
                    best_children = [child]
                elif child_value == best_value:
                    best_children.append(child)
                beta = min(beta, child_value)

            if beta <= alpha:
                break  # Pruning

        # Choose a random child among those with the best value
        best_child = random.choice(best_children) if best_children else None
        
        # Save result in the transposition table
        result = (best_value, best_child)
        self.transposition_table[state_key] = result

        if self.starting_L == L:
            self.starting_L = None

        return best_value, best_child



    def pred_blminmax(self, state):
        if not self.model:
            self.model = joblib.load('model_L4_l1.pkl')
        self.maximizing_player = True
        
        children = self.get_children(state)
        
        if not children:
            return self.H_0(state), state
        
        child_values = []
        for child in children:
            import pandas as pd
            child_value = self.model.predict(pd.DataFrame([[self.H_0(child)]], columns=['evaluation']))[0]
            child_values.append((child, child_value))
        
        
        # Find the child and its value with the maximum predicted value
        best_child, best_value = max(child_values, key=lambda x: x[1])
            
        return best_value, best_child  # Return both the value and the corresponding child



    def mi_pred_blminmax(self, state, features, features_names):
        if not self.model:
            self.model = joblib.load('mlp_model_L4_l1.pkl')
        
        self.maximizing_player = True
        
        children = self.get_children(state)
        
        if not children:
            return self.H_0(state), state
        
        child_values = []
        for child in children:
            import pandas as pd
            child_value = self.model.predict(pd.DataFrame([features], columns=features_names))[0]
            child_values.append((child, child_value))
            
        
        # Find the child and its value with the maximum predicted value
        best_child, best_value = max(child_values, key=lambda x: x[1])
        
        return best_value, best_child