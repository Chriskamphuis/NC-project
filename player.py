from game import *
from random import randint

import numpy as np

class Player():

    def __init__(self, game, value):
        self.game = game
        self.value = value
    
        # sign into game    
        game.set_player(self)
        
    # Get a move (standard random move)
    def get_move(self, legal_moves):
        choice = randint(0, len(legal_moves)-1)
        
        return legal_moves[choice]
        
    def tell_outcome (self, board, score):
        continue


class EndStatePlayer(Player):
    
    # Translates a board into a three-dimensional input array for the neural nets
    # Here dimension 1 is always empty, 2 own moves, 3 opponents' moves
    def board_2_input (self, board):
    
        board = self.game.get_board()
        shape = board.shape
        
        # Create input shape
        input_arr = np.zeros((shape[0], shape[1], 3), dtype=np.int8)
        
        # Fill each dimension
        input_arr[:,:,0] = (board == 0).astype(int)
        input_arr[:,:,1] = (board == self.value).astype(int)
        input_arr[:,:,2] = input_arr[:,:,0] == input_arr[:,:,1]
        
        return input_arr       
        

    def get_move(self, legal_moves):
        
        #for move in legal_moves:
        #    utility = forward_pass through network
        # pick highest utility
        
        choice = randint(0, len(legal_moves)-1)
        
        return legal_moves[choice]
        
    def tell_outcome (self, board, score):
    
        input_arr = self.board_2_input(board)
    
        # pred = forward_pass(input_arr)
        # Train network on board and score
        # backprop(pred, score)
    
        continue
