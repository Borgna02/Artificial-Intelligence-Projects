from collections import deque
from typing import Literal
from minmax import MinMax, Algorithms
MAX_NUM_OF_PIECES = 16

class Player:
    def __init__(self, chess, color: Literal["white", "black"], algorithm: Algorithms, heuristic: Literal["standard", "better"] = "standard", statistics_mode=False):
        self.color = color
        self.chess = chess
        
        # Dictionary to store the move to get a certain board
        self.move_to_the_board = {}
        
        # Deque to store last 16 moves
        self.last_moves = deque(maxlen=16)

        # Dynamically assign evaluate based on heuristic
        self.evaluate = lambda board: getattr(self, f"{heuristic}_evaluate_chessboard")(
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
        
        self.engine = MinMax(self.get_children, self.evaluate, algorithm)
        self.choose_move = self.engine.engine
        
        
        # Statistics
        if statistics_mode:
            self.completed_moves_per_npieces = {i: 0 for i in range (1, MAX_NUM_OF_PIECES + 1)} # Numero di mosse per numero di pezzi
            self.move_times_per_npieces = {i: 0 for i in range(1, MAX_NUM_OF_PIECES + 1)}
            
            self.completed_moves_per_nmoves = {} # Numero di mosse per numero di mosse possibili
            self.move_times_per_nmoves = {}
            
    def register_statistics(self,
        n_pieces: int,
        n_moves: int,
        elapsed_time: float,
    ):
       
        # Register statistics
        self.completed_moves_per_npieces[n_pieces] = self.completed_moves_per_npieces.get(n_pieces, 0) + 1
        self.completed_moves_per_nmoves[n_moves] = self.completed_moves_per_nmoves.get(n_moves, 0) + 1

        self.move_times_per_npieces[n_pieces] = self.move_times_per_npieces.get(n_pieces, 0.0) + elapsed_time
        self.move_times_per_nmoves[n_moves] = self.move_times_per_nmoves.get(n_moves, 0.0) + elapsed_time


    
    def standard_evaluate_chessboard(self, player_color: Literal["white", "black"], board):
        """
        Heuristic evaluation function for chess based on material advantage.

        Args:
            board (dict): Configurazione corrente della scacchiera.
            player_color (str): Colore del giocatore corrente ("white" o "black").

        Returns:
            int: Punteggio della scacchiera per il giocatore specificato.
        """
        PIECE_VALUES = {
            "pawn": 1,
            "knight": 3,
            "bishop": 3,
            "rook": 5,
            "queen": 9,
            "king": 0
        }

        pieces = self.chess.get_available_pieces(board)
        score = 0

        for piece in pieces["white"]:
            piece_name = piece["piece_name"]
            if player_color == "white":
                score += PIECE_VALUES[piece_name]
            else:
                score -= PIECE_VALUES[piece_name]

        for piece in pieces["black"]:
            piece_name = piece["piece_name"]
            if player_color == "black":
                score += PIECE_VALUES[piece_name]
            else:
                score -= PIECE_VALUES[piece_name]

        return score

    def better_evaluate_chessboard(self, player_color:Literal["white", "black"], board):
        """
        Heuristic evaluation function for chess based on material advantage,
        piece mobility, control of the center, and king safety.

        Args:
            board (dict): Configurazione corrente della scacchiera.
            player_color (str): Colore del giocatore corrente ("white" o "black").

        Returns:
            int: Punteggio della scacchiera per il giocatore specificato.
        """
        PIECE_VALUES = {
            "pawn": 1,
            "knight": 3,
            "bishop": 3,
            "rook": 5,
            "queen": 9,
            "king": 0
        }

        # Central squares (e4, d4, e5, d5)
        CENTER_SQUARES = [(3, 3), (3, 4), (4, 3), (4, 4)]
        pieces = self.chess.get_available_pieces(board)

        score = 0

        # Valutazione dei pezzi bianchi
        for piece in pieces["white"]:
            piece_name = piece["piece_name"]
            piece_coordinates = piece["piece_coordinates"]

            # Material advantage
            score += PIECE_VALUES[piece_name] if player_color == "white" else - \
                PIECE_VALUES[piece_name]

            # Bonus for controlling the center
            if tuple(piece_coordinates) in CENTER_SQUARES:
                score += 0.5 if player_color == "white" else -0.5

            # Bonus for piece mobility
            moves = self.chess.possible_moves(
                f"white_{piece_name}", piece_coordinates, board)
            # 0.1 point for each possible move
            mobility_score = len(moves) * 0.1
            score += mobility_score if player_color == "white" else -mobility_score

        # Valutazione dei pezzi neri
        for piece in pieces["black"]:
            piece_name = piece["piece_name"]
            piece_coordinates = piece["piece_coordinates"]

            # Material advantage
            score += PIECE_VALUES[piece_name] if player_color == "black" else - \
                PIECE_VALUES[piece_name]

            # Bonus for controlling the center
            if tuple(piece_coordinates) in CENTER_SQUARES:
                score += 0.5 if player_color == "black" else -0.5

            # Bonus for piece mobility
            moves = self.chess.possible_moves(
                f"black_{piece_name}", piece_coordinates, board)
            # 0.1 point for each possible move
            mobility_score = len(moves) * 0.1
            score += mobility_score if player_color == "black" else -mobility_score

        # Penalizzazione per i re esposti
        try:
            white_king = next(
                piece for piece in pieces["white"] if piece["piece_name"] == "king")
            white_king_safety = self.evaluate_king_safety(
                white_king["piece_coordinates"], board, player_color)
        except StopIteration:
            # Il re bianco è stato mangiato, punteggio minimo
            return float('-inf') if player_color == "white" else float('inf')

        try:
            black_king = next(
                piece for piece in pieces["black"] if piece["piece_name"] == "king")
            black_king_safety = self.evaluate_king_safety(
                black_king["piece_coordinates"], board, player_color)
        except StopIteration:
            # Il re nero è stato mangiato, punteggio minimo
            return float('-inf') if player_color == "black" else float('inf')

        # Considera la sicurezza del re nel punteggio finale
        if player_color == "white":
            score += white_king_safety - black_king_safety
        else:
            score += black_king_safety - white_king_safety

        return score

    def evaluate_king_safety(self, king_position, board, player_color):
        """
        Evaluates the safety of a king based on its surroundings.
        """
        safe_squares = []
        opponent_color = "white" if player_color == "black" else "black"

        # Offset per le caselle adiacenti al re (raggio 1)
        adjacent_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),         (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]

        # Ottieni tutte le mosse possibili dei pezzi avversari
        opponent_moves = set()
        
        moves = self.chess.get_all_possible_moves(opponent_color, board)
        
        # for piece in self.chess.get_available_pieces(board)[opponent_color]:
        #     piece_name = f"{opponent_color}_{piece['piece_name']}"
        #     piece_moves = self.chess.possible_moves(
        #         piece_name, piece["piece_coordinates"], board)

        #     # Converti ogni mossa in una tupla prima di aggiungerla al set
        for piece_moves in moves.values():
            for move in piece_moves:
                opponent_moves.add(tuple(move))

        # Verifica le caselle attorno al re
        for offset in adjacent_offsets:
            x, y = king_position[0] + offset[0], king_position[1] + offset[1]

            # Assicurati che la casella sia valida e all'interno dei limiti della scacchiera
            if 0 <= x < 8 and 0 <= y < 8:
                if (x, y) not in opponent_moves:  # Controlla se la casella è sicura
                    safe_squares.append([x, y])

        return len(safe_squares) * -0.2  # Penalize for fewer safe squares
    
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
            self.last_moves[i + offset] == self.last_moves[i + offset + cycle_length]
            for i in range(cycle_length)
        )