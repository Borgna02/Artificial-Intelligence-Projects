from game import Game
from minmax import Algorithms
from player import Player

game = Game(in_jupiter=False)
opponent = Player(color="black", algorithm=Algorithms.MULTI_INPUT_PRED_BLMINMAX, engine_params={}, heuristic="SEF", statistics_mode=False)
opponent2 = Player(color="white", algorithm=Algorithms.BRANCHING_LIMIT, engine_params={"L":4, "BF":2, "l":1}, heuristic="SEF", statistics_mode=False)
game.start_game(
    ai_players={"black": opponent, "white": opponent2})






