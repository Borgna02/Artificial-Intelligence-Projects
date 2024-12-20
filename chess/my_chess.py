import pygame
from pygame.locals import *
import random

from minmax import MinMax, Algorithms
from piece import Piece
from utils import Utils
from player import Player

import time
from collections import deque

import json


class Chess(object):
    def __init__(self, screen, pieces_src, square_coords, square_length, random_configuration_steps: int, ai_players: dict, statistic_mode=False):

        self.statistic_mode = statistic_mode
        # Inizializza una deque per tracciare le ultime mosse

        # display surface
        self.screen = screen
        # create an object of class to show chess pieces on the board
        self.chess_pieces = Piece(pieces_src, cols=6, rows=2)
        # store coordinates of the chess board squares
        self.board_locations = square_coords
        # length of the side of a chess board square
        self.square_length = square_length
            
        # dictionary to keeping track of player turn
        self.turn = {"black": 0,
                     "white": 0}

        # define engines for the AI players
        self.num_players = 2 - len(ai_players)
        match self.num_players:
            case 0:
                
                self.white_player = ai_players["white"]
                self.white_player.set_chess(self)
                
                # self.white_player = Player(self, "white", Algorithms.BRANCHING_LIMIT, "standard", False)
                self.black_player = ai_players["black"]
                self.black_player.set_chess(self)
                
            case 1:
                if list(ai_players.keys())[0] == "black":
                    self.black_player = ai_players["black"]
                    self.black_player.set_chess(self)
                    self.white_player = None
                else:
                    self.white_player = ai_players["white"]
                    self.white_player.set_chess(self)
                    self.black_player = None
            case 2:
                self.black_player = None
                self.white_player = None

        # list containing possible moves for the selected piece
        self.moves = []
        #
        self.utils = Utils()

        # mapping of piece names to index of list containing piece coordinates on spritesheet
        self.pieces = {
            "white_pawn":   5,
            "white_knight": 3,
            "white_bishop": 2,
            "white_rook":   4,
            "white_king":   0,
            "white_queen":  1,
            "black_pawn":   11,
            "black_knight": 9,
            "black_bishop": 8,
            "black_rook":   10,
            "black_king":   6,
            "black_queen":  7
        }

        # list containing captured pieces
        self.captured = []
        #
        self.winner = ""

        self.reset()
        if random_configuration_steps > 0:
            self.random_configuration(random_configuration_steps)

    def reset(self):
        self.winner = ""

        # clear moves lists
        self.moves = []

        # randomize player turn
        x = random.randint(0, 1)
        if (x == 1):
            self.turn["black"] = 1
        elif (x == 0):
            self.turn["white"] = 1

        # two dimensonal dictionary containing details about each board location
        # storage format is [piece_name, currently_selected, x_y_coordinate]
        self.piece_location = {}
        x = 0
        for i in range(97, 105):
            a = 8
            y = 0
            self.piece_location[chr(i)] = {}
            while a > 0:
                # [piece name, currently selected, board coordinates]
                self.piece_location[chr(i)][a] = ["", False, [x, y]]
                a = a - 1
                y = y + 1
            x = x + 1

        # reset the board
        for i in range(97, 105):
            x = 8
            while x > 0:
                if (x == 8):
                    if (chr(i) == 'a' or chr(i) == 'h'):
                        self.piece_location[chr(i)][x][0] = "black_rook"
                    elif (chr(i) == 'b' or chr(i) == 'g'):
                        self.piece_location[chr(i)][x][0] = "black_knight"
                    elif (chr(i) == 'c' or chr(i) == 'f'):
                        self.piece_location[chr(i)][x][0] = "black_bishop"
                    elif (chr(i) == 'd'):
                        self.piece_location[chr(i)][x][0] = "black_queen"
                    elif (chr(i) == 'e'):
                        self.piece_location[chr(i)][x][0] = "black_king"
                elif (x == 7):
                    self.piece_location[chr(i)][x][0] = "black_pawn"
                elif (x == 2):
                    self.piece_location[chr(i)][x][0] = "white_pawn"
                elif (x == 1):
                    if (chr(i) == 'a' or chr(i) == 'h'):
                        self.piece_location[chr(i)][x][0] = "white_rook"
                    elif (chr(i) == 'b' or chr(i) == 'g'):
                        self.piece_location[chr(i)][x][0] = "white_knight"
                    elif (chr(i) == 'c' or chr(i) == 'f'):
                        self.piece_location[chr(i)][x][0] = "white_bishop"
                    elif (chr(i) == 'd'):
                        self.piece_location[chr(i)][x][0] = "white_queen"
                    elif (chr(i) == 'e'):
                        self.piece_location[chr(i)][x][0] = "white_king"
                x = x - 1

    #

    def play_turn(self):
        # white color
        white_color = (255, 255, 255)
        # create fonts for texts
        small_font = pygame.font.SysFont("comicsansms", 20)
        # create text to be shown on the game menu

        if self.turn["black"]:
            turn_text = small_font.render("Turn: Black", True, white_color)
        elif self.turn["white"]:
            turn_text = small_font.render("Turn: White", True, white_color)

        # show welcome text
        self.screen.blit(turn_text,
                            ((self.screen.get_width() - turn_text.get_width()) // 2,
                            10))

        number_of_black_pieces = len(self.get_available_pieces()["black"])
        number_of_possible_moves = sum(len(self.get_all_possible_moves("black")[i]) for i in self.get_all_possible_moves("black").keys())
        
        
        # let player with black piece play
        destination_move = None
        if (self.turn["black"]):
            
            if self.black_player:
                start = time.time()
                _, board = self.black_player.choose_move(
                    self.piece_location,
                )
                end = time.time()
                elapsed_time = end - start
                
                if self.statistic_mode:
                    self.black_player.register_statistics(number_of_black_pieces, number_of_possible_moves, elapsed_time)

                if not board:
                    self.winner = "Empty l189"

                    return

                try:
                    piece_name, piece_color, (source_coordinates), (
                        destination_coordinates) = self.black_player.move_to_the_board[json.dumps(board)]
                    self.black_player.move_to_the_board.clear()
                except KeyError:
                    self.winner = "Empty l198"
                    return
                    
                source_move = (piece_name, piece_color, source_coordinates)
                destination_move = ("", "", destination_coordinates)

                # print("Thinking...")
                self.move_piece("black", source_move)

            self.move_piece("black", destination_move)

        # let player with white piece play
        elif (self.turn["white"]):
            if self.white_player:

                _, board = self.white_player.choose_move(
                    self.piece_location,
                )

                if not board:
                    self.winner = "Empty l218"
                    return

                try:
                    piece_name, piece_color, (source_coordinates), (
                        destination_coordinates) = self.white_player.move_to_the_board[json.dumps(board)]
                    self.white_player.move_to_the_board.clear()
                except KeyError:
                    self.winner = "Empty l226"
                    return

                source_move = (piece_name, piece_color, source_coordinates)
                destination_move = ("", "", destination_coordinates)

                self.move_piece("white", source_move)

            self.move_piece("white", destination_move)

        if self.black_player and self.turn["black"]:
            # Aggiungi le mosse
            self.black_player.last_moves.append((source_move, destination_move))

            # Controlla se c'è un ciclo
            if any(self.black_player.detect_cycle(cycle_length=i) for i in range(2, 16)):
                self.winner = "Draw"
                # method to draw pieces on the chess board
        
        if self.white_player and self.turn["white"]:
            # Aggiungi le mosse
            self.white_player.last_moves.append((source_move, destination_move))

            # Controlla se c'è un ciclo
            if any(self.white_player.detect_cycle(cycle_length=i) for i in range(2, 16)):
                self.winner = "Draw"
    
    def draw_pieces(self):
        transparent_green = (0, 194, 39, 170)
        transparent_blue = (28, 21, 212, 170)

        # create a transparent surface
        surface = pygame.Surface(
            (self.square_length, self.square_length), pygame.SRCALPHA)
        surface.fill(transparent_green)

        surface1 = pygame.Surface(
            (self.square_length, self.square_length), pygame.SRCALPHA)
        surface1.fill(transparent_blue)

        # loop to change background color of selected piece
        for val in self.piece_location.values():
            for value in val.values():
                # name of the piece in the current location
                piece_name = value[0]
                # x, y coordinates of the current piece
                piece_coord_x, piece_coord_y = value[2]

                # change background color of piece if it is selected
                if value[1] and len(value[0]) > 5:
                    # if the piece selected is a black piece
                    if value[0][:5] == "black":
                        self.screen.blit(
                            surface, self.board_locations[piece_coord_x][piece_coord_y])
                        if len(self.moves) > 0:
                            for move in self.moves:
                                x_coord = move[0]
                                y_coord = move[1]
                                if x_coord >= 0 and y_coord >= 0 and x_coord < 8 and y_coord < 8:
                                    self.screen.blit(
                                        surface, self.board_locations[x_coord][y_coord])
                    # if the piece selected is a white piece
                    elif value[0][:5] == "white":
                        self.screen.blit(
                            surface1, self.board_locations[piece_coord_x][piece_coord_y])
                        if len(self.moves) > 0:
                            for move in self.moves:
                                x_coord = move[0]
                                y_coord = move[1]
                                if x_coord >= 0 and y_coord >= 0 and x_coord < 8 and y_coord < 8:
                                    self.screen.blit(
                                        surface1, self.board_locations[x_coord][y_coord])

        # draw all chess pieces
        for val in self.piece_location.values():
            for value in val.values():
                # name of the piece in the current location
                piece_name = value[0]
                # x, y coordinates of the current piece
                piece_coord_x, piece_coord_y = value[2]
                # check if there is a piece at the square
                if (len(value[0]) > 1):
                    # draw piece on the board
                    self.chess_pieces.draw(self.screen, piece_name,
                                           self.board_locations[piece_coord_x][piece_coord_y])

    # method to find the possible moves of the selected piece

    def possible_moves(self, piece_name, piece_coord, piece_locations=None):

        if not piece_locations:
            piece_locations = self.piece_location

        # list to store possible moves of the selected piece
        positions = []
        # find the possible locations to put a piece
        if len(piece_name) > 0:
            # get x, y coordinate
            x_coord, y_coord = piece_coord
            # calculate moves for bishop
            if piece_name[6:] == "bishop":
                positions = self.diagonal_moves(
                    positions, piece_name, piece_coord)

            # calculate moves for pawn
            elif piece_name[6:] == "pawn":
                # convert list index to dictionary key
                columnChar = chr(97 + x_coord)
                rowNo = 8 - y_coord

                # calculate moves for white pawn
                if piece_name == "black_pawn":
                    if y_coord + 1 < 8:
                        # get row in front of black pawn
                        rowNo = rowNo - 1
                        front_piece = self.piece_location[columnChar][rowNo][0]

                        # pawns cannot move when blocked by another another pawn
                        if (front_piece[6:] != "pawn"):
                            positions.append([x_coord, y_coord+1])
                            # black pawns can move two positions ahead for first move
                            if y_coord < 2:
                                positions.append([x_coord, y_coord+2])

                        # EM PASSANT
                        # diagonal to the left
                        if x_coord - 1 >= 0 and y_coord + 1 < 8:
                            x = x_coord - 1
                            y = y_coord + 1

                            # convert list index to dictionary key
                            columnChar = chr(97 + x)
                            rowNo = 8 - y
                            to_capture = self.piece_location[columnChar][rowNo]

                            if (to_capture[0][:5] == "white"):
                                positions.append([x, y])

                        # diagonal to the right
                        if x_coord + 1 < 8 and y_coord + 1 < 8:
                            x = x_coord + 1
                            y = y_coord + 1

                            # convert list index to dictionary key
                            columnChar = chr(97 + x)
                            rowNo = 8 - y
                            to_capture = self.piece_location[columnChar][rowNo]

                            if (to_capture[0][:5] == "white"):
                                positions.append([x, y])

                # calculate moves for white pawn
                elif piece_name == "white_pawn":
                    if y_coord - 1 >= 0:
                        # get row in front of black pawn
                        rowNo = rowNo + 1
                        front_piece = self.piece_location[columnChar][rowNo][0]

                        # pawns cannot move when blocked by another another pawn
                        if (front_piece[6:] != "pawn"):
                            positions.append([x_coord, y_coord-1])
                            # black pawns can move two positions ahead for first move
                            if y_coord > 5:
                                positions.append([x_coord, y_coord-2])

                        # EM PASSANT
                        # diagonal to the left
                        if x_coord - 1 >= 0 and y_coord - 1 >= 0:
                            x = x_coord - 1
                            y = y_coord - 1

                            # convert list index to dictionary key
                            columnChar = chr(97 + x)
                            rowNo = 8 - y
                            to_capture = self.piece_location[columnChar][rowNo]

                            if (to_capture[0][:5] == "black"):
                                positions.append([x, y])

                        # diagonal to the right
                        if x_coord + 1 < 8 and y_coord - 1 >= 0:
                            x = x_coord + 1
                            y = y_coord - 1

                            # convert list index to dictionary key
                            columnChar = chr(97 + x)
                            rowNo = 8 - y
                            to_capture = self.piece_location[columnChar][rowNo]

                            if (to_capture[0][:5] == "black"):
                                positions.append([x, y])

            # calculate moves for rook
            elif piece_name[6:] == "rook":
                # find linear moves
                positions = self.linear_moves(
                    positions, piece_name, piece_coord)

            # calculate moves for knight
            elif piece_name[6:] == "knight":
                # left positions
                if (x_coord - 2) >= 0:
                    if (y_coord - 1) >= 0:
                        positions.append([x_coord-2, y_coord-1])
                    if (y_coord + 1) < 8:
                        positions.append([x_coord-2, y_coord+1])
                # top positions
                if (y_coord - 2) >= 0:
                    if (x_coord - 1) >= 0:
                        positions.append([x_coord-1, y_coord-2])
                    if (x_coord + 1) < 8:
                        positions.append([x_coord+1, y_coord-2])
                # right positions
                if (x_coord + 2) < 8:
                    if (y_coord - 1) >= 0:
                        positions.append([x_coord+2, y_coord-1])
                    if (y_coord + 1) < 8:
                        positions.append([x_coord+2, y_coord+1])
                # bottom positions
                if (y_coord + 2) < 8:
                    if (x_coord - 1) >= 0:
                        positions.append([x_coord-1, y_coord+2])
                    if (x_coord + 1) < 8:
                        positions.append([x_coord+1, y_coord+2])

            # calculate movs for king
            elif piece_name[6:] == "king":
                if (y_coord - 1) >= 0:
                    # top spot
                    positions.append([x_coord, y_coord-1])

                if (y_coord + 1) < 8:
                    # bottom spot
                    positions.append([x_coord, y_coord+1])

                if (x_coord - 1) >= 0:
                    # left spot
                    positions.append([x_coord-1, y_coord])
                    # top left spot
                    if (y_coord - 1) >= 0:
                        positions.append([x_coord-1, y_coord-1])
                    # bottom left spot
                    if (y_coord + 1) < 8:
                        positions.append([x_coord-1, y_coord+1])

                if (x_coord + 1) < 8:
                    # right spot
                    positions.append([x_coord+1, y_coord])
                    # top right spot
                    if (y_coord - 1) >= 0:
                        positions.append([x_coord+1, y_coord-1])
                    # bottom right spot
                    if (y_coord + 1) < 8:
                        positions.append([x_coord+1, y_coord+1])

            # calculate movs for queen
            elif piece_name[6:] == "queen":
                # find diagonal positions
                positions = self.diagonal_moves(
                    positions, piece_name, piece_coord)

                # find linear moves
                positions = self.linear_moves(
                    positions, piece_name, piece_coord)

            # list of positions to be removed
            to_remove = []

            # remove positions that overlap other pieces of the current player
            for pos in positions:
                x, y = pos

                # convert list index to dictionary key
                columnChar = chr(97 + x)
                rowNo = 8 - y

                # find the pieces to remove
                des_piece_name = self.piece_location[columnChar][rowNo][0]
                if (des_piece_name[:5] == piece_name[:5]):
                    to_remove.append(pos)

            # remove position from positions list
            for i in to_remove:
                positions.remove(i)

        # return list containing possible moves for the selected piece
        return positions

    def move_piece(self, turn, selected_move=None):
        # get the coordinates of the square selected on the board
        if not selected_move:
            square = self.get_selected_square()

            # if a square was selected
            if square:
                # get name of piece on the selected square
                piece_name = square[0]
                # color of piece on the selected square
                piece_color = piece_name[:5]
                # board column character
                columnChar = square[1]
                # board row number
                rowNo = square[2]

                # get x, y coordinates
                x, y = self.piece_location[columnChar][rowNo][2]
        else:
            piece_name, piece_color, (columnChar, rowNo) = selected_move
            x, y = self.piece_location[columnChar][rowNo][2]  # Coordinate x, y

        if selected_move or not selected_move and square:
            # if there's a piece on the selected square
            if (len(piece_name) > 0) and (piece_color == turn):
                # find possible moves for thr piece
                self.moves = self.possible_moves(piece_name, [x, y])

            # checkmate mechanism
            p = self.piece_location[columnChar][rowNo]

            for i in self.moves:
                if i == [x, y]:
                    if (p[0][:5] == turn) or len(p[0]) == 0:
                        self.validate_move([x, y])
                    else:
                        self.capture_piece([columnChar, rowNo], [x, y])

            # only the player with the turn gets to play
            if (piece_color == turn):
                # change selection flag from all other pieces
                for k in self.piece_location.keys():
                    for key in self.piece_location[k].keys():
                        self.piece_location[k][key][1] = False

                # change selection flag of the selected piece
                self.piece_location[columnChar][rowNo][1] = True

    def simulate_move(self, piece_name, piece_coordinates, move, current_board):
        # Create a copy of the board to preserve the original state
        new_board = {col: {row: cell.copy() for row, cell in rows.items()}
                    for col, rows in current_board.items()}

        # Locate the current position of the piece based on its coordinates
        current_position = None
        for column, rows in current_board.items():
            for row, cell in rows.items():
                if cell[0] == piece_name and cell[2] == piece_coordinates:
                    current_position = (column, row)
                    break
            if current_position:
                break

        if not current_position:
            raise ValueError(f"Piece {piece_name} with coordinates {piece_coordinates} is not on the board.")

        # Find the new position based on the move coordinates
        new_position = None
        for column, rows in current_board.items():
            for row, cell in rows.items():
                if cell[2] == move:
                    new_position = (column, row)
                    break
            if new_position:
                break

        if not new_position:
            raise ValueError(f"Move {move} is invalid.")

        # Update the piece's position
        current_column, current_row = current_position
        new_column, new_row = new_position

        # Remove the piece from its current position
        new_board[current_column][current_row][0] = ''
        new_board[current_column][current_row][1] = False

        # Place the piece in its new position
        new_board[new_column][new_row][0] = piece_name
        new_board[new_column][new_row][1] = True

        return new_board

    def get_selected_square(self):
        # get left event
        left_click = self.utils.left_click_event()
        # if there's a mouse event
        if left_click:
            # get mouse event
            mouse_event = self.utils.get_mouse_event()

            for i in range(len(self.board_locations)):
                for j in range(len(self.board_locations)):
                    rect = pygame.Rect(self.board_locations[i][j][0], self.board_locations[i][j][1],
                                       self.square_length, self.square_length)
                    collision = rect.collidepoint(
                        mouse_event[0], mouse_event[1])
                    if collision:
                        selected = [rect.x, rect.y]
                        # find x, y coordinates the selected square
                        for k in range(len(self.board_locations)):
                            #
                            try:
                                l = None
                                l = self.board_locations[k].index(selected)
                                if l != None:
                                    # reset color of all selected pieces
                                    for val in self.piece_location.values():
                                        for value in val.values():
                                            # [piece name, currently selected, board coordinates]
                                            if not value[1]:
                                                value[1] = False

                                    # get column character and row number of the chess piece
                                    columnChar = chr(97 + k)
                                    rowNo = 8 - l
                                    # get the name of the
                                    piece_name = self.piece_location[columnChar][rowNo][0]

                                    return [piece_name, columnChar, rowNo]
                            except:
                                pass
        else:
            return None

    def capture_piece(self, chess_board_coord, piece_coord):
        

        # get chess board coordinate
        columnChar, rowNo = chess_board_coord

        p = self.piece_location[columnChar][rowNo]

        if p[0] == "white_king":
            self.winner = "Black"
            # print("Black wins")
        elif p[0] == "black_king":
            self.winner = "White"
            # print("White wins")

        # add the captured piece to list
        self.captured.append(p)
        # move source piece to its destination
        self.validate_move(piece_coord)

    def validate_move(self, destination):
        desColChar = chr(97 + destination[0])
        desRowNo = 8 - destination[1]

        for k in self.piece_location.keys():
            for key in self.piece_location[k].keys():
                board_piece = self.piece_location[k][key]

                if board_piece[1]:
                    # unselect the source piece
                    self.piece_location[k][key][1] = False
                    # get the name of the source piece
                    piece_name = self.piece_location[k][key][0]
                    # move the source piece to the destination piece
                    self.piece_location[desColChar][desRowNo][0] = piece_name

                    src_name = self.piece_location[k][key][0]
                    # remove source piece from its current position
                    self.piece_location[k][key][0] = ""

                    # change turn
                    if (self.turn["black"]):
                        self.turn["black"] = 0
                        self.turn["white"] = 1
                    elif ("white"):
                        self.turn["black"] = 1
                        self.turn["white"] = 0

                    src_location = k + str(key)
                    des_location = desColChar + str(desRowNo)
                    # print("{} moved from {} to {}".format(
                    #     src_name,  src_location, des_location))

    # helper function to find diagonal moves

    def diagonal_moves(self, positions, piece_name, piece_coord, piece_locations=None):

        if not piece_locations:
            piece_locations = self.piece_location

        # reset x and y coordinate values
        x, y = piece_coord
        # find top left diagonal spots
        while (True):
            x = x - 1
            y = y - 1
            if (x < 0 or y < 0):
                break
            else:
                positions.append([x, y])

            # convert list index to dictionary key
            columnChar = chr(97 + x)
            rowNo = 8 - y
            p = piece_locations[columnChar][rowNo]

            # stop finding possible moves if blocked by a piece
            if len(p[0]) > 0 and piece_name[:5] != p[:5]:
                break

        # reset x and y coordinate values
        x, y = piece_coord
        # find bottom right diagonal spots
        while (True):
            x = x + 1
            y = y + 1
            if (x > 7 or y > 7):
                break
            else:
                positions.append([x, y])

            # convert list index to dictionary key
            columnChar = chr(97 + x)
            rowNo = 8 - y
            p = piece_locations[columnChar][rowNo]

            # stop finding possible moves if blocked by a piece
            if len(p[0]) > 0 and piece_name[:5] != p[:5]:
                break

        # reset x and y coordinate values
        x, y = piece_coord
        # find bottom left diagonal spots
        while (True):
            x = x - 1
            y = y + 1
            if (x < 0 or y > 7):
                break
            else:
                positions.append([x, y])

            # convert list index to dictionary key
            columnChar = chr(97 + x)
            rowNo = 8 - y
            p = piece_locations[columnChar][rowNo]

            # stop finding possible moves if blocked by a piece
            if len(p[0]) > 0 and piece_name[:5] != p[:5]:
                break

        # reset x and y coordinate values
        x, y = piece_coord
        # find top right diagonal spots
        while (True):
            x = x + 1
            y = y - 1
            if (x > 7 or y < 0):
                break
            else:
                positions.append([x, y])

            # convert list index to dictionary key
            columnChar = chr(97 + x)
            rowNo = 8 - y
            p = self.piece_location[columnChar][rowNo]

            # stop finding possible moves if blocked by a piece
            if len(p[0]) > 0 and piece_name[:5] != p[:5]:
                break

        return positions

    # helper function to find horizontal and vertical moves

    def linear_moves(self, positions, piece_name, piece_coord, piece_locations=None):

        if not piece_locations:
            piece_locations = self.piece_location

        # reset x, y coordniate value
        x, y = piece_coord
        # horizontal moves to the left
        while (x > 0):
            x = x - 1
            positions.append([x, y])

            # convert list index to dictionary key
            columnChar = chr(97 + x)
            rowNo = 8 - y
            p = piece_locations[columnChar][rowNo]

            # stop finding possible moves if blocked by a piece
            if len(p[0]) > 0 and piece_name[:5] != p[:5]:
                break

        # reset x, y coordniate value
        x, y = piece_coord
        # horizontal moves to the right
        while (x < 7):
            x = x + 1
            positions.append([x, y])

            # convert list index to dictionary key
            columnChar = chr(97 + x)
            rowNo = 8 - y
            p = piece_locations[columnChar][rowNo]

            # stop finding possible moves if blocked by a piece
            if len(p[0]) > 0 and piece_name[:5] != p[:5]:
                break

        # reset x, y coordniate value
        x, y = piece_coord
        # vertical moves upwards
        while (y > 0):
            y = y - 1
            positions.append([x, y])

            # convert list index to dictionary key
            columnChar = chr(97 + x)
            rowNo = 8 - y
            p = piece_locations[columnChar][rowNo]

            # stop finding possible moves if blocked by a piece
            if len(p[0]) > 0 and piece_name[:5] != p[:5]:
                break

        # reset x, y coordniate value
        x, y = piece_coord
        # vertical moves downwards
        while (y < 7):
            y = y + 1
            positions.append([x, y])

            # convert list index to dictionary key
            columnChar = chr(97 + x)
            rowNo = 8 - y
            p = piece_locations[columnChar][rowNo]

            # stop finding possible moves if blocked by a piece
            if len(p[0]) > 0 and piece_name[:5] != p[:5]:
                break

        return positions

    def get_available_pieces(self, piece_locations=None):

        if not piece_locations:
            piece_locations = self.piece_location

        pieces = {"white": [], "black": []}
        for k in piece_locations.keys():
            for key in piece_locations[k].keys():
                # Return only the piece name
                board_piece = piece_locations[k][key]
                if len(board_piece[0]) > 0:
                    piece_color = board_piece[0][:5]
                    piece_name = board_piece[0][6:]
                    piece_coordinates = board_piece[2]

                    piece_dict = {"piece_name": piece_name,
                                  "piece_coordinates": piece_coordinates}
                    if piece_color == "white":
                        pieces["white"].append(piece_dict)
                    elif piece_color == "black":
                        pieces["black"].append(piece_dict)

        return pieces

    def get_all_possible_moves(self, player_color, current_board=None):
        
        pieces = self.get_available_pieces(current_board)
        
        moves = {}
        for piece in pieces[player_color]:
            piece_coordinates = piece["piece_coordinates"]
            piece_name = f"{player_color}_{piece['piece_name']}"
            
            moves[f"{piece_name}_{piece_coordinates}"] = self.possible_moves(
                piece_name, piece_coordinates, current_board)
            
        return moves

    def get_children_boards(self, player_color, current_board=None):
        """
        Genera tutte le configurazioni figlie per un dato giocatore.

        Args:
            player_color (str): Colore del giocatore corrente ("white" o "black").
            current_board (dict): Configurazione corrente della scacchiera.

        Returns:
            list: Lista delle configurazioni di scacchiera dopo ogni mossa possibile.
        """

        pieces = self.get_available_pieces(current_board)
        moves = self.get_all_possible_moves(player_color, current_board)
        
        children = []  # Lista per i nodi figli

        for piece in pieces[player_color]:
            piece_coordinates = piece["piece_coordinates"]
            piece_name = f"{player_color}_{piece['piece_name']}"

            for move in moves[f"{piece_name}_{piece_coordinates}"]:
                # Simula la mossa
                board = self.simulate_move(
                    piece_name, piece_coordinates, move, current_board)

                children.append(board)

                # Salva il mapping della mossa
                mapping = (piece_name, player_color,
                           (chr(piece_coordinates[0] + 97),
                            8 - piece_coordinates[1]),
                           (chr(move[0] + 97), 8 - move[1]))

                if player_color == "black":
                    self.black_player.move_to_the_board[json.dumps(board)] = mapping
                elif self.num_players == 0 and player_color == "white":
                    self.white_player.move_to_the_board[json.dumps(board)] = mapping

        return children

    def board_to_text(self, board):
        """
        Converte una scacchiera in formato JSON in una rappresentazione testuale con simboli.

        Args:
            board (dict): La scacchiera in formato JSON.

        Returns:
            str: La scacchiera in formato testuale con simboli Unicode.
        """
        # Mappa dei simboli Unicode per i pezzi
        piece_symbols = {
            "white_king": "♔", "black_king": "♚",
            "white_queen": "♕", "black_queen": "♛",
            "white_rook": "♖", "black_rook": "♜",
            "white_bishop": "♗", "black_bishop": "♝",
            "white_knight": "♘", "black_knight": "♞",
            "white_pawn": "♙", "black_pawn": "♟"
        }

        # Mappa per le colonne (a-h)
        columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        text_board = ""

        for row in range(8, 0, -1):  # Riga da 8 a 1
            line = ""
            for col in columns:
                piece = board[col][row][0]
                if piece:
                    # Usa "?" per pezzi sconosciuti
                    line += piece_symbols.get(piece, "?")
                else:
                    line += "."  # Casella vuota
                line += " "
            text_board += line.strip() + "\n"

        return text_board.strip()

 
    

    def random_configuration(self, number_of_moves):
        """
        Genera una configurazione casuale della scacchiera dopo un numero di mosse specificato.

        Args:
            number_of_moves (int): Numero di mosse casuali da eseguire.

        Returns:
            dict: Configurazione casuale della scacchiera.
        """
        current_board = self.piece_location
        player_color = "white"

        for _ in range(number_of_moves):
            children = self.get_children_boards(player_color, current_board)
            current_board = random.choice(children)
            # print("Mossa casuale:\n", self.board_to_text(current_board), "\n")
            player_color = "black" if player_color == "white" else "white"

        self.piece_location = current_board


    
    def is_square_attacked(self, square, attacker_color, board):
        """
        Determina se una casella è attaccata da un pezzo di un certo colore.
        
        Args:
            square (tuple): Coordinata della casella (x, y).
            attacker_color (str): Colore degli attaccanti ("white" o "black").
            board: La configurazione corrente della scacchiera.
        
        Returns:
            bool: True se la casella è attaccata, False altrimenti.
        """
        # Ottieni tutte le mosse possibili per i pezzi del colore attaccante
        possible_moves = self.get_all_possible_moves(attacker_color, board)
        
        # Controlla se la casella è inclusa in una delle mosse
        for moves in possible_moves.values():
            if square in moves:
                return True
        return False
    
    def determine_phase(self):
        """
        Determina la fase della partita basandosi sul materiale e la posizione dei pezzi.
        
        Args:
            board: Configurazione corrente della scacchiera.

        Returns:
            str: "opening", "middlegame", o "endgame".
        """

        # Definizione dei valori dei pezzi (escludendo il re)
        piece_values = {
            "pawn": 1,
            "knight": 3,
            "bishop": 3,
            "rook": 5,
            "queen": 9,
            "king": 0  # Il re è escluso dal calcolo del materiale
        }

        # Ottieni i pezzi disponibili
        available_pieces = self.get_available_pieces()

        # Calcola il materiale totale
        total_material = sum(
            piece_values[piece["piece_name"]]
            for color in available_pieces
            for piece in available_pieces[color]
            if piece["piece_name"] != "king"  # Escludi il re
        )

        # Determina la fase della partita in base al materiale totale
        if total_material > 20:
            return "opening"
        elif total_material > 10:
            return "middlegame"
        else:
            return "endgame"