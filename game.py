import numpy as np
import random

from player import *

class Game:

    def __init__(self, playerOne, playerTwo, board=None):
        
        # Define players
        self.playerOne = playerOne
        self.playerTwo = playerTwo

        # Define the board
        self.board = board
        if board is None:
            self.board = np.zeros((6, 7), dtype=np.int8)
        (self.height, self.width) = self.board.shape
            
    # Function to switch players (who goes first)
    def switch_players(self):
        temp = self.playerOne
        self.playerOne = self.playerTwo
        self.playerTwo = temp
        
    # Function that returns a list of legal moves in the current board
    def get_legal_moves(self):
        legal_moves = list()
        for move in range(0, self.width):
            played = sum([1 for e in self.board[:, move] if e != 0])
            if (played < self.height):
                legal_moves.append(move)
        return legal_moves
  
    # Function to play a single game. Player that starts is given as argument
    # training is a boolean function
    def play_game(self, training):
        
        player = self.playerOne
        winner = 0
        moves = 0

        while(winner == 0 and self.not_full()):
            # Decide legal moves and shuffle them (for variety)
            legal_moves = self.get_legal_moves()
            np.random.shuffle(legal_moves)

            # Request move from player
            move = player.get_move(self, legal_moves, training)
            
            winner = self.play_move(move, player.value)

            #if(training==False):
                #self.print_board()
            
            if (player == self.playerOne):
                player = self.playerTwo
            else:
                player = self.playerOne

            moves += 1

        return (winner, moves)
        
    # Check if the board is full, for draws
    def not_full(self):
        return not self.board.all()
  
    # Function to apply a move to the board
    def play_move(self, move, value):
        played = sum([1 for e in self.board[:, move] if e != 0])
        
        if played > self.height-1:
            raise ValueError('The column is already full')
        
        self.board[self.height-1-played, move] = value
        
        if (self.end_state(move, self.height-1-played, value)):
            return value
        else:
            return 0
    
    # Boolean function to check if the game is done
    def end_state(self, col, row, value):
        # Check for horizontal
        if (self.check_neighbour(col, row, value, 1, 0) + self.check_neighbour(col-1, row, value, -1, 0) >= 4):
            return True
            
        # Check for vertical   
        elif (self.check_neighbour(col, row, value, 0, 1) + self.check_neighbour(col, row-1, value, 0, -1) >= 4):
            return True
        
        # Check for upwards diagonal
        elif (self.check_neighbour(col, row, value, -1, -1) + self.check_neighbour(col+1, row+1, value, 1, 1) >= 4):
            return True
            
        # Check for downwards diagonal
        elif (self.check_neighbour(col, row, value, -1, 1) + self.check_neighbour(col+1, row-1, value, 1, -1) >= 4):
            return True
        
        # No connect-four found     
        return False    
    
    # Recursive function to check if neighbour in direction (dx,dy) has the same value    
    def check_neighbour(self, col, row, value, dx, dy):
        if (col < 0 or col >= self.width or row < 0 or row >= self.height):
            return 0
    
        elif (self.board[row, col] == value):
            return 1 + self.check_neighbour(col+dx, row+dy, value, dx, dy)
            
        else:
            return 0
            
    # Function that resets the board
    def reset_board(self):
        #self.board = self.board * 0
        self.board = np.zeros((6, 7), dtype=np.int8)

    # Function to print current board
    def print_board(self, winner):
        print("printing board")
        print(self.board)
        print("Player {0} won!".format(winner))
      
      
    ###########################################################################  
    # Plays a random game from a given board position, then returns the score #
    ###########################################################################
    
    def sample_game(self, board, last_player, nr_samples):
    
        # Save current board to memory
        current_board = self.board.copy()
        old_playerOne = self.playerOne
        old_playerTwo = self.playerTwo

        # Determine starting random player from given position        
        val_one = self.playerOne.value
        val_two = self.playerTwo.value

        if (last_player == self.playerOne):
            self.playerOne = Player(val_two)
            self.playerTwo = Player(val_one)
        else:
            self.playerOne = Player(val_one)
            self.playerTwo = Player(val_two) 

        # Play nr_samples random games
        score = 0.0
        for i in range(nr_samples):
        
            # Set input board a start point
            self.board = board.copy()

            # Play random game
            winner = self.play_game(False)

            # Player who played last move before sampling won
            if (winner[0] == last_player.value): 
                score += 1.0

            # Game was a draw
            elif (winner[0] == 0):
                score += 0.5

            # Player who did not play last move before sampling won        
            else:
                score += 0.0

        # Place original settings back
        self.board = current_board
        self.playerOne = old_playerOne
        self.playerTwo = old_playerTwo
        
        # Return final score
        return score/nr_samples
        
        
