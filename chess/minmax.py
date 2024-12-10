

from typing import Literal
from enum import Enum

class Algorithms(Enum):
    ALPHA_BETA = "alpha_beta"
    BRANCHING_LIMIT = "branching_limit"
    
    
    
class MinMax:


    def __init__(self, get_children, evaluate, engine: Algorithms = Algorithms.ALPHA_BETA):
        self.maximizing_player = None
        self.get_children = get_children
        self.evaluate = evaluate
        self.operation_count = 0

        match engine:
            case Algorithms.ALPHA_BETA:
                self.engine = self.abminmax
            case Algorithms.BRANCHING_LIMIT:
                self.engine = self.blminmax
            case _:
                raise ValueError(
                    "Invalid engine type. Choose between 'alpha_beta' and 'branching_limit'.")

    def abminmax(self, node, depth, maximizing_player, alpha=float('-inf'), beta=float('inf')):
        """
        Implementazione generica dell'algoritmo MinMax con potatura Alpha-Beta.

        Args:
            node: il nodo corrente dello stato di gioco.
            depth: profondità massima per cui eseguire l'algoritmo.
            alpha: il valore Alpha (la migliore opzione già trovata per il giocatore massimizzante).
            beta: il valore Beta (la migliore opzione già trovata per il giocatore minimizzante).
            maximizing_player: booleano, True se il giocatore corrente è il massimizzante, False altrimenti.
            get_children: funzione che restituisce i nodi figli del nodo corrente.
            evaluate: funzione di valutazione che restituisce un punteggio per un nodo terminale o uno stato di gioco.

        Returns:
            Il miglior valore possibile per il giocatore corrente.
        """
        self.maximizing_player = maximizing_player
        get_children = self.get_children
        evaluate = self.evaluate
        
        # Incrementa il contatore delle operazioni
        self.operation_count += 1
        
        # Condizione terminale: profondità zero o stato terminale
        if depth == 0 or not get_children(node):
            return evaluate(node), node

        # Inizializza i valori di confronto e il nodo migliore
        best_value = float('-inf') if maximizing_player else float('inf')
        best_child = None

        # Ciclo sui figli
        for child in get_children(node):
            eval, _ = self.abminmax(
                child, depth - 1, not maximizing_player, alpha, beta)

            # Aggiorna il miglior valore e il nodo migliore
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

            # Potatura
            if beta <= alpha:
                break

        return best_value, best_child

    def blminmax(self, node, depth, maximizing_player, alpha=float('-inf'), beta=float('inf'), branching_factor=5):
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
        self.maximizing_player = maximizing_player
        get_children = self.get_children
        evaluate = self.evaluate
        
        # Incrementa il contatore delle operazioni
        self.operation_count += 1
        
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

    def reset_operation_count(self):
        """Resetta il contatore delle operazioni."""
        self.operation_count = 0