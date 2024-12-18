from collections import deque
from typing import Literal
from minmax import MinMax, Algorithms
import time
MAX_NUM_OF_PIECES = 16


class Player:
    def __init__(self, color: Literal["white", "black"], algorithm: Algorithms, engine_params: dict, strong_level: int, statistics_mode=False):
        self.color = color
        self.strong_level = strong_level
        self.algorithm = algorithm
        self.engine_params = engine_params

        # Dictionary to store the move to get a certain board
        self.move_to_the_board = {}

        # Deque to store last 16 moves
        self.last_moves = deque(maxlen=16)

        # Statistics
        self.statistics_mode = statistics_mode
        if statistics_mode:
            self.completed_moves_per_npieces = {i: 0 for i in range(
                1, MAX_NUM_OF_PIECES + 1)}  # Numero di mosse per numero di pezzi
            self.move_times_per_npieces = {
                i: 0 for i in range(1, MAX_NUM_OF_PIECES + 1)}

            # Numero di mosse per numero di mosse possibili
            self.completed_moves_per_nmoves = {}
            self.move_times_per_nmoves = {}

    def set_chess(self, chess):
        self.chess = chess

        # Dynamically assign evaluate based on heuristic
        self.evaluate = lambda board: self.combined_evaluation(
            self.color if self.engine.maximizing_player else (
                "black" if self.color == "white" else "white"),
            board,
        )

        # Get children lambda function
        self.get_children = lambda board: self.chess.get_children_boards(
            self.color if self.engine.maximizing_player else (
                "black" if self.color == "white" else "white"),
            board
        )

        self.engine = MinMax(self.get_children, self.evaluate, self.algorithm)

    def choose_move(self, board):
        
        print
        best_value, best_board = self.engine.engine(board, **self.engine_params) 

        if self.statistics_mode:
            # Save the tuple H_0, H_L
            with open('dataset.csv', 'a') as f:
                f.write(f"{self.evaluate(board)},{best_value}\n")

        return best_value, best_board

    def detect_cycle(self, cycle_length):
        """
        Rileva se gli ultimi movimenti contengono un ciclo della lunghezza specificata.

        :param last_moves: Deque contenente le ultime mosse.
        :param cycle_length: Lunghezza del ciclo da verificare.
        :return: True se viene rilevato un ciclo, False altrimenti.
        """
        # Controlla se ci sono abbastanza mosse per verificare un ciclo
        if len(self.last_moves) < cycle_length * 2:
            return False

        # Confronta la prima metà con la seconda metà
        moves_len = len(self.last_moves)
        offset = moves_len - cycle_length * 2

        # Confronta la prima metà con la seconda metà
        return all(
            self.last_moves[i + offset] == self.last_moves[i +
                                                           offset + cycle_length]
            for i in range(cycle_length)
        )

    def register_statistics(self,
                            n_pieces: int,
                            n_moves: int,
                            elapsed_time: float,
                            ):

        # Register statistics
        self.completed_moves_per_npieces[n_pieces] = self.completed_moves_per_npieces.get(
            n_pieces, 0) + 1
        self.completed_moves_per_nmoves[n_moves] = self.completed_moves_per_nmoves.get(
            n_moves, 0) + 1

        self.move_times_per_npieces[n_pieces] = self.move_times_per_npieces.get(
            n_pieces, 0.0) + elapsed_time
        self.move_times_per_nmoves[n_moves] = self.move_times_per_nmoves.get(
            n_moves, 0.0) + elapsed_time

    def evaluate_material(self, player_color: Literal["white", "black"], board):
        """
        Valuta il vantaggio materiale del giocatore.
        """
        PIECE_VALUES = {"pawn": 1, "knight": 3,
                        "bishop": 3, "rook": 5, "queen": 9, "king": 0}
        pieces = self.chess.get_available_pieces(board)
        score = 0

        for piece in pieces["white"]:
            value = PIECE_VALUES[piece["piece_name"]]
            score += value if player_color == "white" else -value

        for piece in pieces["black"]:
            value = PIECE_VALUES[piece["piece_name"]]
            score += value if player_color == "black" else -value

        return score

    def simplified_evaluation(self, player_color: Literal["white", "black"], board):
        """
        Implementa la Simplified Evaluation Function (SEF) per valutare la scacchiera.

        Args:
            player_color (Literal["white", "black"]): Colore del giocatore valutato.
            board: Lo stato corrente della scacchiera.

        Returns:
            float: Valore della valutazione della scacchiera.
        """

        # Materiale (M)
        def material_value():
            piece_values = {"pawn": 1, "knight": 3,
                            "bishop": 3, "rook": 5, "queen": 9, "king": 0}
            pieces = self.chess.get_available_pieces(board)
            score = 0
            for piece in pieces[player_color]:
                score += piece_values[piece["piece_name"]]
            for piece in pieces["black" if player_color == "white" else "white"]:
                score -= piece_values[piece["piece_name"]]
            return score

        # Penalità per struttura pedonale (P)
        def pawn_structure_penalty():
            penalty = 0
            pawns = [
                piece for piece in self.chess.get_available_pieces(board)[player_color]
                if piece["piece_name"] == "pawn"
            ]

            # Identifica i pedoni doppiati, isolati, arretrati
            pawn_files = [pawn["piece_coordinates"][1] for pawn in pawns]
            unique_files = set(pawn_files)

            for file in unique_files:
                count = pawn_files.count(file)
                if count > 1:  # Pedoni doppiati
                    penalty -= (count - 1) * 0.5

                # Penalità per pedoni isolati (nessun pedone nelle colonne adiacenti)
                if not any(f in pawn_files for f in [file - 1, file + 1]):
                    penalty -= 0.5

            return penalty

        # Controllo del centro (C)
        def center_control():
            center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]
            control_score = 0
            pieces = self.chess.get_available_pieces(board)[player_color]

            for piece in pieces:
                if tuple(piece["piece_coordinates"]) in center_squares:
                    control_score += 0.5

            return control_score

        # Mobilità (Mv)
        def mobility():
            moves = self.chess.get_all_possible_moves(player_color, board)
            return len(moves) * 0.1  # Ogni mossa vale 0.1 punti

        # Sicurezza del re (Ks)
        def king_safety():
            try:
                king = next(
                    piece for piece in self.chess.get_available_pieces(board)[player_color]
                    if piece["piece_name"] == "king"
                )
            except StopIteration:
                # Penalità massima se il re non è presente
                return float('-100')

            adjacent_offsets = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),         (0, 1),
                (1, -1), (1, 0), (1, 1)
            ]
            safe_squares = 0

            for offset in adjacent_offsets:
                x, y = king["piece_coordinates"][0] + \
                    offset[0], king["piece_coordinates"][1] + offset[1]
                if 0 <= x < 8 and 0 <= y < 8:  # Dentro i limiti della scacchiera
                    if not self.chess.is_square_attacked((x, y), "black" if player_color == "white" else "white", board):
                        safe_squares += 1

            return safe_squares * 0.2  # Ogni casella sicura vale 0.2 punti

        # Calcola la valutazione combinata
        phase = self.chess.determine_phase()

        # Pesi dinamici in base alla fase del gioco
        phase_weights = {
            "opening": {"material": 1.0, "pawn_structure": 0.2, "center_control": 0.5, "mobility": 0.5, "king_safety": 0.1},
            "middlegame": {"material": 1.0, "pawn_structure": 0.3, "center_control": 0.3, "mobility": 0.4, "king_safety": 0.3},
            "endgame": {"material": 1.5, "pawn_structure": 0.2, "center_control": 0.1, "mobility": 0.2, "king_safety": 0.5},
        }

        weights = phase_weights[phase]

        # Componenti della valutazione
        material = material_value() * weights["material"]
        pawn_structure = pawn_structure_penalty() * weights["pawn_structure"]
        center_control_val = center_control() * weights["center_control"]
        mobility_val = mobility() * weights["mobility"]
        king_safety_val = king_safety() * weights["king_safety"]

        # Calcolo del punteggio totale
        score = material + pawn_structure + center_control_val + mobility_val + king_safety_val

        return score

    def combined_evaluation(self, player_color: Literal["white", "black"], board):
        """
        Combina le valutazioni euristiche in base a pesi bilanciati e normalizza nell'intervallo [0, 10].
        Il risultato dipende dallo strong_level.
        """

        # Valutazione semplificata
        simplified_value = self.simplified_evaluation(player_color, board)

        # Valutazione materiale
        material_value = self.evaluate_material(player_color, board)

        return material_value if self.strong_level == 0 else simplified_value
