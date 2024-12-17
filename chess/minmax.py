

import time
from typing import Callable
from enum import Enum

class Algorithms(Enum):
    MIN_MAX = "min_max"
    FAIL_HARD_ALPHA_BETA = "fail_hard_alpha_beta"
    FAIL_SOFT_ALPHA_BETA = "fail_soft_alpha_beta"
    BRANCHING_LIMIT = "branching_limit"
    
    
class MinMax:
    def __init__(self, get_children: Callable, evaluate:Callable, engine: Algorithms):
        self.maximizing_player = None
        self.get_children = get_children
        self.H_0 = evaluate

        match engine:
            case Algorithms.MIN_MAX:
                self.engine = self.minmax
            case Algorithms.FAIL_HARD_ALPHA_BETA:
                self.engine = self.fhabminmax
            case Algorithms.FAIL_SOFT_ALPHA_BETA:
                self.engine = self.fsabminmax
            case Algorithms.BRANCHING_LIMIT:
                self.engine = self.blminmax
            case _:
                raise ValueError(
                    F"Invalid engine type. Choose between {', '.join([engine.value for engine in Algorithms])}")
                
    def minmax(self, state, L, maximizing_player=True):

        # Aggiorna il giocatore corrente
        self.maximizing_player = maximizing_player

        # Memorizza i figli per evitare calcoli ripetuti
        children = self.get_children(state)

        # Condizione terminale: profondità zero o stato terminale
        if L == 0 or not children:
            return self.H_0(state), state

        # Inizializza il valore migliore e il nodo migliore
        best_value = float('-inf') if maximizing_player else float('inf')
        best_move = None

        # Ciclo sui figli
        for next_state in children:
            current_value, _ = self.minmax(next_state, L - 1, not maximizing_player)

            if maximizing_player:
                if current_value > best_value:
                    best_value = current_value
                    best_move = next_state
            else:
                if current_value < best_value:
                    best_value = current_value
                    best_move = next_state

        return best_value, best_move


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
        best_child = None

        # Ciclo sui figli
        for child in children:
            child_value, _ = self.fhabminmax(child, L - 1, alpha, beta, not maximizing_player)

            if maximizing_player:
                if child_value > best_value:
                    best_value = child_value
                    best_child = child
                alpha = max(alpha, best_value)
            else:
                if child_value < best_value:
                    best_value = child_value
                    best_child = child
                beta = min(beta, best_value)
            if beta < alpha:
                break  # Potatura Beta

        return best_value, best_child

    def fsabminmax(self, state, L, alpha=float('-inf'), beta=float('inf'), maximizing_player=True):
  
        # Aggiorna il giocatore corrente
        self.maximizing_player = maximizing_player
        get_children = self.get_children
        evaluate = self.H_0
        

        # Condizione terminale: profondità zero o stato terminale
        if L == 0 or not get_children(state):
            return evaluate(state), state
        
        # Inizializza il valore migliore e il nodo migliore
        best_value = float('-inf') if maximizing_player else float('inf')
        best_child = None

        # Ciclo sui figli
        for child in get_children(state):
            child_value, _ = self.fsabminmax(child, L - 1, alpha, beta, not maximizing_player)

            if maximizing_player:
                if child_value > best_value:
                    best_value = child_value
                    best_child = child
                alpha = max(alpha, best_value)
            else:
                if child_value < best_value:
                    best_value = child_value
                    best_child = child
                beta = min(beta, best_value)
            if beta <= alpha:
                break  # Potatura Beta

        return best_value, best_child

    

    def blminmax(self, state, L, branching_factor=5, alpha=float('-inf'), beta=float('inf'), maximizing_player=True):
        """
        Implementazione di MinMax con potatura Alpha-Beta e limite sul branching factor.

        Args:
            state: Stato corrente del gioco.
            L: Profondità massima.
            branching_factor: Numero massimo di figli da esplorare.
            alpha: Valore Alpha per potatura.
            beta: Valore Beta per potatura.
            maximizing_player: True se è il turno del giocatore massimizzante.

        Returns:
            tuple: (miglior valore, miglior stato figlio)
        """
        self.maximizing_player = maximizing_player
        get_children = self.get_children
        evaluate = self.H_0
        
        # Condizione terminale
        children = get_children(state)
        if L == 0 or not children:
            return evaluate(state), state

        best_value = float('-inf') if maximizing_player else float('inf')
        best_child = None

        # Valuta i figli
        evaluated_children = [(child, evaluate(child)) for child in children]

        # Ordina i figli e limita il numero
        evaluated_children.sort(key=lambda x: x[1], reverse=maximizing_player)
        
        # Se il numero di figli è minore del branching factor, esplora tutti i figli
        limited_children = evaluated_children[:min(branching_factor, len(children))]


        # Ciclo sui figli limitati
        for child, _ in limited_children:
            child_value, _ = self.blminmax(child, L - 1, branching_factor, alpha, beta, not maximizing_player)
            
            if maximizing_player:
                if child_value > best_value:
                    best_value = child_value
                    best_child = child
                alpha = max(alpha, child_value)
            else:
                if child_value < best_value:
                    best_value = child_value
                    best_child = child
                beta = min(beta, child_value)
            
            if beta <= alpha:
                break  # Potatura


        # Se best_child non è stato aggiornato, assegna un figlio di default
        if best_child is None and limited_children:
            time.sleep(1000 * 60 * 30)
            best_child = limited_children[0][0]  # Primo figlio valido

        return best_value, best_child


