import os
from typing import Literal
import pygame
from pygame.locals import *
from tqdm import tqdm
from minmax import Algorithms
from my_chess import Chess
from player import MAX_NUM_OF_PIECES
from utils import Utils
import matplotlib.pyplot as plt
from IPython import get_ipython


class Game:
    def __init__(self, in_jupiter: bool = False):
           # Check if the environment is Jupyter
        self.in_jupyter = in_jupiter and get_ipython() is not None

        if self.in_jupyter:
            # Use a virtual display for Pygame in Jupyter
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        # screen dimensions
        screen_width = 640
        screen_height = 750
        # flag to know if game menu has been showed
        self.menu_showed = False
        # flag to set game loop
        self.running = True
        # base folder for program resources
        self.resources = "res"

        # initialize game window
        pygame.display.init()
        # initialize font for text
        pygame.font.init()

        # create game window
        self.screen = pygame.display.set_mode([screen_width, screen_height])

        # title of window
        window_title = "Chess"
        # set window caption
        pygame.display.set_caption(window_title)

        # get location of game icon
        icon_src = os.path.join(self.resources, "chess_icon.png")
        # load game icon
        icon = pygame.image.load(icon_src)
        # set game icon
        pygame.display.set_icon(icon)
        # update display
        pygame.display.flip()
        # set game clock
        self.clock = pygame.time.Clock()

    def start_game(self, ai_players: dict, random_configuration_steps: int = 0):
        """Function containing main game loop"""
        # chess board offset
        self.board_offset_x = 0
        self.board_offset_y = 50
        self.board_dimensions = (self.board_offset_x, self.board_offset_y)

        # get location of chess board image
        board_src = os.path.join(self.resources, "board.png")
        # load the chess board image
        self.board_img = pygame.image.load(board_src).convert()

        # get the width of a chess board square
        square_length = self.board_img.get_rect().width // 8

        # initialize list that stores all places to put chess pieces on the board
        self.board_locations = []

        # calculate coordinates of the each square on the board
        for x in range(0, 8):
            self.board_locations.append([])
            for y in range(0, 8):
                self.board_locations[x].append([self.board_offset_x+(x*square_length),
                                                self.board_offset_y+(y*square_length)])

        # get location of image containing the chess pieces
        pieces_src = os.path.join(self.resources, "pieces.png")
        # create class object that handles the gameplay logic
        self.chess = Chess(self.screen, pieces_src, self.board_locations, square_length,
                           random_configuration_steps, ai_players)

        # game loop
        while self.running:
            self.clock.tick(5)
            # poll events
            for event in pygame.event.get():
                # get keys pressed
                key_pressed = pygame.key.get_pressed()
                # check if the game has been closed by the user
                if event.type == pygame.QUIT or key_pressed[K_ESCAPE]:
                    # set flag to break out of the game loop
                    self.running = False
                elif key_pressed[K_SPACE]:
                    self.chess.reset()

            winner = self.chess.winner

            if self.menu_showed == False:
                self.menu()
            elif len(winner) > 0:
                self.declare_winner(winner)
            else:
                self.game()

            # update display
            pygame.display.flip()
            # update events
            pygame.event.pump()

        # call method to stop pygame
        pygame.quit()

    def menu(self):
        """method to show game menu"""
        # background color
        bg_color = (255, 255, 255)
        # set background color
        self.screen.fill(bg_color)
        # black color
        black_color = (0, 0, 0)
        # coordinates for "Play" button
        start_btn = pygame.Rect(270, 300, 100, 50)
        # show play button
        pygame.draw.rect(self.screen, black_color, start_btn)

        # white color
        white_color = (255, 255, 255)
        # create fonts for texts
        big_font = pygame.font.SysFont("comicsansms", 50)
        small_font = pygame.font.SysFont("comicsansms", 20)
        # create text to be shown on the game menu
        welcome_text = big_font.render("Chess", False, black_color)
        created_by = small_font.render("Created by Sheriff", True, black_color)
        start_btn_label = small_font.render("Play", True, white_color)

        # show welcome text
        self.screen.blit(welcome_text,
                         ((self.screen.get_width() - welcome_text.get_width()) // 2,
                          150))
        # show credit text
        self.screen.blit(created_by,
                         ((self.screen.get_width() - created_by.get_width()) // 2,
                          self.screen.get_height() - created_by.get_height() - 100))
        # show text on the Play button
        self.screen.blit(start_btn_label,
                         ((start_btn.x + (start_btn.width - start_btn_label.get_width()) // 2,
                           start_btn.y + (start_btn.height - start_btn_label.get_height()) // 2)))

        # get pressed keys
        key_pressed = pygame.key.get_pressed()
        #
        util = Utils()

        # check if left mouse button was clicked
        if util.left_click_event():
            # call function to get mouse event
            mouse_coords = util.get_mouse_event()

            # check if "Play" button was clicked
            if start_btn.collidepoint(mouse_coords[0], mouse_coords[1]):
                # change button behavior as it is hovered
                pygame.draw.rect(self.screen, white_color, start_btn, 3)

                # change menu flag
                self.menu_showed = True
            # check if enter or return key was pressed
            elif key_pressed[K_RETURN]:
                self.menu_showed = True

    def game(self):

        # background color
        color = (0, 0, 0)
        # set backgound color
        self.screen.fill(color)

        # show the chess board
        self.screen.blit(self.board_img, self.board_dimensions)

        # call self.chess. something
        self.chess.play_turn()

        # draw pieces on the chess board
        self.chess.draw_pieces()

    def declare_winner(self, winner):
        # background color
        bg_color = (255, 255, 255)
        # set background color
        self.screen.fill(bg_color)
        # black color
        black_color = (0, 0, 0)
        # coordinates for play again button
        reset_btn = pygame.Rect(250, 300, 140, 50)
        # show reset button
        pygame.draw.rect(self.screen, black_color, reset_btn)

        # white color
        white_color = (255, 255, 255)
        # create fonts for texts
        big_font = pygame.font.SysFont("comicsansms", 50)
        small_font = pygame.font.SysFont("comicsansms", 20)

        # text to show winner
        text = winner + " wins!"
        winner_text = big_font.render(text, False, black_color)

        # create text to be shown on the reset button
        reset_label = "Play Again"
        reset_btn_label = small_font.render(reset_label, True, white_color)

        # show winner text
        self.screen.blit(winner_text,
                         ((self.screen.get_width() - winner_text.get_width()) // 2,
                          150))

        # show text on the reset button
        self.screen.blit(reset_btn_label,
                         ((reset_btn.x + (reset_btn.width - reset_btn_label.get_width()) // 2,
                           reset_btn.y + (reset_btn.height - reset_btn_label.get_height()) // 2)))

        # get pressed keys
        key_pressed = pygame.key.get_pressed()
        #
        util = Utils()

        # check if left mouse button was clicked
        if util.left_click_event():
            # call function to get mouse event
            mouse_coords = util.get_mouse_event()

            # check if reset button was clicked
            if reset_btn.collidepoint(mouse_coords[0], mouse_coords[1]):
                # change button behavior as it is hovered
                pygame.draw.rect(self.screen, white_color, reset_btn, 3)

                # change menu flag
                self.menu_showed = False
            # check if enter or return key was pressed
            elif key_pressed[K_RETURN]:
                self.menu_showed = False
            # reset game
            self.chess.reset()
            # clear winner
            self.chess.winner = ""

    # The evaluated player is the black player
    def start_in_statistics_mode(self, ai_players: dict, nruns=20, random_configuration_steps: int = 10, plot: bool = True):

        # To not display the game in jupyter

        wins = 0
        completed_runs = nruns
        failed_runs = 0
        draws = 0
        # Numero di mosse per numero di pezzi
        completed_moves_per_npieces = {
            i: 0 for i in range(1, MAX_NUM_OF_PIECES + 1)}
        move_times_per_npieces = {
            i: 0 for i in range(1, MAX_NUM_OF_PIECES + 1)}

        completed_moves_per_nmoves = {}
        move_times_per_nmoves = {}

        for _ in tqdm(range(nruns), desc="Running games"):
            # create class object that handles the gameplay logic
            # chess board offset
            self.board_offset_x = 0
            self.board_offset_y = 20
            self.board_dimensions = (
                self.board_offset_x, self.board_offset_y)

            # get location of chess board image
            board_src = os.path.join(self.resources, "board.png")
            # load the chess board image
            self.board_img = pygame.image.load(board_src).convert()

            # get the width of a chess board square
            square_length = self.board_img.get_rect().width // 8

            # initialize list that stores all places to put chess pieces on the board
            self.board_locations = []

            # calculate coordinates of the each square on the board
            for x in range(0, 8):
                self.board_locations.append([])
                for y in range(0, 8):
                    self.board_locations[x].append([self.board_offset_x+(x*square_length),
                                                    self.board_offset_y+(y*square_length)])

            # get location of image containing the chess pieces
            pieces_src = os.path.join(self.resources, "pieces.png")
            # create class object that handles the gameplay logic

            self.chess = Chess(self.screen, pieces_src, self.board_locations, square_length,
                               random_configuration_steps, ai_players, statistic_mode=True)
            while len(self.chess.winner) == 0:

                # background color
                color = (0, 0, 0)
                # set backgound color
                self.screen.fill(color)

                # show the chess board
                self.screen.blit(self.board_img, self.board_dimensions)

                self.chess.play_turn()

                self.chess.draw_pieces()

                pygame.display.flip()
                # update events
                pygame.event.pump()
                
            if self.chess.winner == "Black":
                wins += 1
            elif self.chess.winner == "Draw":
                completed_runs -= 1
                draws += 1
            elif self.chess.winner == "Empty":
                completed_runs -= 1
                failed_runs += 1

            for i in range(1, MAX_NUM_OF_PIECES + 1):
                completed_moves_per_npieces[i] += self.chess.black_player.completed_moves_per_npieces[i]
                move_times_per_npieces[i] += self.chess.black_player.move_times_per_npieces[i]
                
            
            for i in self.chess.black_player.completed_moves_per_nmoves.keys():
                if i not in completed_moves_per_nmoves:
                    completed_moves_per_nmoves[i] = 0
                    move_times_per_nmoves[i] = 0
                completed_moves_per_nmoves[i] += self.chess.black_player.completed_moves_per_nmoves[i]
                move_times_per_nmoves[i] += self.chess.black_player.move_times_per_nmoves[i]

            # print(f"\r{self.chess.winner} Wins")

        # call method to stop pygame
        pygame.quit()

        average_move_times_per_npieces = {i: move_times_per_npieces[i] / completed_moves_per_npieces[i] if completed_moves_per_npieces[i] != 0 else 0
                              for i in range(1, MAX_NUM_OF_PIECES + 1)}
        win_rate = wins/completed_runs if completed_runs != 0 else 0
        
        
        average_move_times_per_nmoves = {i: move_times_per_nmoves[i] / completed_moves_per_nmoves[i] if completed_moves_per_nmoves[i] != 0 else 0 for i in sorted(move_times_per_nmoves.keys())}
        
        ai_players["black"].win_rate = win_rate
        ai_players["black"].wins = wins
        ai_players["black"].draws = draws
        ai_players["black"].failed_runs = failed_runs
        ai_players["black"].completed_runs = completed_runs
        

        if plot:
            print(f"Average moves times per piece: {average_move_times_per_npieces}")
            print(f"Win rate: {win_rate}, Wins: {wins}, Draws: {draws}, Runs: {nruns}, Failed: {failed_runs}")

            # Plot average move times
            plt.figure(figsize=(10, 5))
            plt.plot(average_move_times_per_npieces.keys(),
                    average_move_times_per_npieces.values(), color='blue')
            plt.xlabel('Number of Pieces')
            plt.ylabel('Average Move Time (s)')
            plt.title('Average Move Time per Number of Pieces')
            plt.show()


            # Plot average move times
            plt.figure(figsize=(10, 5))
            plt.plot(average_move_times_per_nmoves.keys(),
                    average_move_times_per_nmoves.values(), color='blue')
            plt.xlabel('Number of Possible Moves')
            plt.ylabel('Average Move Time (s)')
            plt.title('Average Move Time per Number of Possible Moves')
            plt.show()
        

            