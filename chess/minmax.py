

from typing import Callable
from enum import Enum

class Algorithms(Enum):
    MIN_MAX = "min_max"
    FAIL_HARD_ALPHA_BETA = "fail_hard_alpha_beta"
    FAIL_SOFT_ALPHA_BETA = "fail_soft_alpha_beta"
    BRANCHING_LIMIT = "branching_limit"
    
    
class MinMax:
    def __init__(self, get_children, evaluate, engine: Algorithms):
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
                
    def minmax(self, state, l, maximizing_player=True):

        # Aggiorna il giocatore corrente
        self.maximizing_player = maximizing_player

        # Memorizza i figli per evitare calcoli ripetuti
        children = self.get_children(state)

        # Condizione terminale: profondità zero o stato terminale
        if l == 0 or not children:
            return self.H_0(state), state

        # Inizializza il valore migliore e il nodo migliore
        best_value = float('-inf') if maximizing_player else float('inf')
        best_move = None

        # Ciclo sui figli
        for next_state in children:
            current_value, _ = self.minmax(next_state, l - 1, not maximizing_player)

            if maximizing_player:
                if current_value > best_value:
                    best_value = current_value
                    best_move = next_state
            else:
                if current_value < best_value:
                    best_value = current_value
                    best_move = next_state

        return best_value, best_move


    def fhabminmax(self, state, l, alpha=float('-inf'), beta=float('inf'), maximizing_player=True):

        # Aggiorna il giocatore corrente
        self.maximizing_player = maximizing_player
        
        # Memorizza i figli per evitare calcoli ripetuti
        children = self.get_children(state)

        # Condizione terminale: profondità zero o stato terminale
        if l == 0 or not children:
            return self.H_0(state), state
        
        # Inizializza il valore migliore e il nodo migliore
        best_value = float('-inf') if maximizing_player else float('inf')
        best_child = None

        # Ciclo sui figli
        for child in children:
            child_value, _ = self.fhabminmax(child, l - 1, alpha, beta, not maximizing_player)

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

    def fsabminmax(self, state, l, alpha=float('-inf'), beta=float('inf'), maximizing_player=True):
  
        # Aggiorna il giocatore corrente
        self.maximizing_player = maximizing_player
        get_children = self.get_children
        evaluate = self.H_0
        
        # Memorizza i figli per evitare calcoli ripetuti
        # children = get_children(state)

        # Condizione terminale: profondità zero o stato terminale
        if l == 0 or not get_children(state):
            return evaluate(state), state
        
        # Inizializza il valore migliore e il nodo migliore
        best_value = float('-inf') if maximizing_player else float('inf')
        best_child = None

        # Ciclo sui figli
        for child in get_children(state):
            child_value, _ = self.fsabminmax(child, l - 1, alpha, beta, not maximizing_player)

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

    

    def blminmax(self, state, l, maximizing_player=True, alpha=float('-inf'), beta=float('inf'), branching_factor=5):
        """
        Implementazione di MinMax con potatura Alpha-Beta e limite sul branching factor.

        Args:
            node: Nodo corrente dello stato di gioco.
            depth: Profondità massima per cui eseguire l'algoritmo.
            alpha: Valore Alpha (massimo valore per il giocatore massimizzante).
            beta: Valore Beta (minimo valore per il giocatore minimizzante).
            maximizing_player: Booleano, True se il giocatore corrente è il massimizzante.
            get_children: Funzione che restituisce i nodi figli del nodo corrente.
            evaluate: Funzione di valutazione che restituisce un punteggio per uno stato di gioco.
            branching_factor: Numero massimo di figli da esplorare per ogni nodo.

        Returns:
            tuple: (miglior valore, miglior stato figlio)
        """
        node = state
        depth = l
        self.maximizing_player = maximizing_player
        get_children = self.get_children
        evaluate = self.H_0
        

        
        if depth == 0 or not get_children(node):
            return evaluate(node), node

        best_value = float('-inf') if maximizing_player else float('inf')
        best_child = None

        # Ottieni i figli e valuta i loro punteggi
        children = get_children(node)
        evaluated_children = [(child, evaluate(child)) for child in children]

        # Ordina i figli in base al punteggio
        evaluated_children.sort(key=lambda x: x[1], reverse=maximizing_player)

        # Limita il numero di figli da esplorare
        limited_children = evaluated_children[:branching_factor]

        # Ciclo sui figli limitati
        for child, _ in limited_children:
            eval, _ = self.blminmax(child, depth - 1,
                                    not maximizing_player, alpha, beta, branching_factor)
            if maximizing_player:
                if eval > best_value:
                    best_value = eval
                    best_child = child
                alpha = max(alpha, eval)
            else:
                if eval < best_value:
                    best_value = eval
                    best_child = child
                beta = min(beta, eval)

            if beta <= alpha:
                break

        return best_value, best_child

