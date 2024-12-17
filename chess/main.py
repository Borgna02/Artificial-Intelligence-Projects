from game import Game
from minmax import Algorithms
from player import Player
if __name__ == "__main__":

    black = Player("black", Algorithms.BRANCHING_LIMIT, {
                   "L": 5, "branching_factor": 1}, 1, True)
    white = Player("white", Algorithms.BRANCHING_LIMIT, {
                   "L": 5, "branching_factor": 1}, 6, True)

    game = Game()
    game.start_game({"black": black, "white": white},
                    random_configuration_steps=20)
    # game.start_in_statistics_mode(nruns=20, evaluated_algorithm=Algorithms.BRANCHING_LIMIT, random_configuration_steps=20, no_display=False)
    # game.start_in_statistics_mode(
    #     nruns=10,
    #     evaluated_algorithm=Algorithms.FAIL_HARD_ALPHA_BETA,
    #     random_configuration_steps=20,
    #     no_display=False
    # )
