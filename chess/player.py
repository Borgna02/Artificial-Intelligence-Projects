from collections import deque
from typing import Literal
from minmax import MinMax, Algorithms
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

        return self.engine.engine(board, **self.engine_params)

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
        PIECE_VALUES = {"pawn": 1, "knight": 3, "bishop": 3, "rook": 5, "queen": 9, "king": 20}
        pieces = self.chess.get_available_pieces(board)
        score = 0

        for piece in pieces["white"]:
            value = PIECE_VALUES[piece["piece_name"]]
            score += value if player_color == "white" else -value

        for piece in pieces["black"]:
            value = PIECE_VALUES[piece["piece_name"]]
            score += value if player_color == "black" else -value

        return score

    def evaluate_mobility(self, player_color: Literal["white", "black"], board):
        """
        Valuta la mobilità dei pezzi (numero di mosse possibili).
        """
        pieces = self.chess.get_available_pieces(board)
        mobility_score = 0

        for piece in pieces[player_color]:
            moves = self.chess.possible_moves(f"{player_color}_{piece['piece_name']}", piece["piece_coordinates"], board)
            mobility_score += len(moves) * 0.1  # Ogni mossa vale 0.1 punti

        return mobility_score

    def evaluate_center_control(self, player_color: Literal["white", "black"], board):
        """
        Valuta il controllo del centro della scacchiera.
        """
        CENTER_SQUARES = {(3, 3), (3, 4), (4, 3), (4, 4)}
        pieces = self.chess.get_available_pieces(board)
        center_score = 0

        for piece in pieces[player_color]:
            if tuple(piece["piece_coordinates"]) in CENTER_SQUARES:
                center_score += 0.5  # Bonus per controllare il centro

        return center_score

    def evaluate_king_safety(self, player_color: Literal["white", "black"], board):
        """
        Valuta la sicurezza del re basandosi sul numero di caselle sicure attorno al re.
        """
        opponent_color = "black" if player_color == "white" else "white"
        adjacent_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),         (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]
        
        # Trova la posizione del re
        pieces = self.chess.get_available_pieces(board)
        try:
            king_position = next(piece["piece_coordinates"] 
                                for piece in pieces[player_color] 
                                if piece["piece_name"] == "king")
        except StopIteration:
            # Se il re è stato catturato, sicurezza minima
            return float('-10')  # Penalità molto alta

        # Ottieni tutte le mosse possibili dei pezzi avversari
        opponent_moves = set()
        moves = self.chess.get_all_possible_moves(opponent_color, board)
        for piece_moves in moves.values():
            for move in piece_moves:
                opponent_moves.add(tuple(move))
        
        # Conta le caselle sicure intorno al re
        safe_squares = 0
        for offset in adjacent_offsets:
            x, y = king_position[0] + offset[0], king_position[1] + offset[1]
            if 0 <= x < 8 and 0 <= y < 8:  # Assicurati che la casella sia valida
                if (x, y) not in opponent_moves:
                    safe_squares += 1  # Casella sicura
        
        # Penalizzazione basata sul numero di caselle sicure
        return safe_squares * -0.2


    def evaluate_opponent_threats(self, player_color: Literal["white", "black"], board):
        """
        Penalizza le minacce dirette ai pezzi del giocatore.
        """
        opponent_color = "black" if player_color == "white" else "white"
        opponent_moves = self.chess.get_all_possible_moves(opponent_color, board)
        threatened_score = 0

        for piece in self.chess.get_available_pieces(board)[player_color]:
            if tuple(piece["piece_coordinates"]) in opponent_moves:
                threatened_score -= 1  # Penalità per pezzi minacciati

        return threatened_score

    def combined_evaluation(self, player_color: Literal["white", "black"], board):
        """
        Combina le valutazioni euristiche in base al livello di forza.
        """
        strong_level = self.strong_level
        score = 0
        if strong_level >= 1:
            score += self.evaluate_material(player_color, board)
        if strong_level >= 2:
            score += self.evaluate_mobility(player_color, board)
        if strong_level >= 3:
            score += self.evaluate_center_control(player_color, board)
        if strong_level >= 4:
            score += self.evaluate_king_safety(player_color, board)
        if strong_level >= 5:
            score += self.evaluate_opponent_threats(player_color, board)
        
        return score