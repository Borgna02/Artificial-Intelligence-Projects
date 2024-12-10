from game import Game
from minmax import Algorithms
if __name__=="__main__":
    game = Game()
    # game.start_game("0_players", random_configuration_steps=20)
    game.start_in_statistics_mode(nruns=20, evaluated_algorithm=Algorithms.BRANCHING_LIMIT, random_configuration_steps=20, no_display=False)
