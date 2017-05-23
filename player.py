from game import *
from random import randint

from network import *

import numpy as np

#################
# RANDOM PLAYER #
#################

class Player():

    def __init__(self, value, sign_in = True):
        self.value = value
        
    # Translates a board into a three-dimensional input array for the neural nets
    # Here dimension 1 is always empty, 2 own moves, 3 opponents' moves
    def board_2_input (self, board):
        shape = board.shape
        
        # Create input shape
        input_arr = np.zeros((1, 3, shape[0], shape[1]), dtype=np.int8)
        
        # Fill each dimension
        input_arr[:,0,:,:] = (board == 0).astype(int)
        input_arr[:,1,:,:] = (board == self.value).astype(int)
        input_arr[:,2,:,:] = input_arr[:,0,:,:] == input_arr[:,1,:,:]
        
        return input_arr
        

    # Get a move (standard random move)
    def get_move(self, board, legal_moves):
        choice = randint(0, len(legal_moves)-1)        
        return legal_moves[choice]
        
    def tell_outcome(self, board, score):
        return ""


####################
# END STATE PLAYER #
####################

class EndStatePlayer(Player):      
        
    def __init__(self, value, network):
        self.value = value
        self.network = network  
        
    # Requests a move from the player, given a board
    def get_move(self, board, legal_moves):
        
        best_move = -1
        best_pred = 0
        
        for move in legal_moves:
            
            # Process move on copy of board
            post_board = board.copy()
            played = sum([1 for e in post_board[:, move] if e != 0])
            post_board[board.shape[0]-1-played, move] = self.value
            
            # Get prediction on post_board
            input_arr = self.board_2_input(post_board)
            pred = self.network.predict(input_arr)
            
            if best_move == -1 or pred > best_pred:
                best_move = move
                best_pred = pred
        
        return best_move
    
    
    # Function to start the training process    
    def tell_outcome(self, board, score):
    
        # Convert board to input shape
        input_arr = self.board_2_input(board)
        
        # Train network on board and score
        self.network.train(input_arr, score)
        
#####################
# Q-LEARNING PLAYER #
#####################

class QLearningPlayer(Player):

    def __init__(self, value, network, discount=0.9):
        self.value = value
        self.network = network
        self.discount = discount
        self.memory = list() 
        
    # Requests a move from the player, given a board
    def get_move(self, board, legal_moves):
        
        # Variables that remember best move data
        best_move = -1
        best_pred = 0
        best_input = None
        
        for move in legal_moves:
            
            # Process move on copy of board
            post_board = board.copy()
            played = sum([1 for e in post_board[:, move] if e != 0])
            post_board[board.shape[0]-1-played, move] = self.value
            
            # Get prediction on post_board
            input_arr = self.board_2_input(board)
            pred = self.network.predict(input_arr)
            
            if best_move == -1 or pred > best_pred:
                best_move = move
                best_pred = pred
                best_input = input_arr
        
        # Add board to memory
        self.memory.append(best_input)
        
        return best_move
    
    
    # Function to start the training process    
    def tell_outcome (self, board, score):
        
        # Get prediction on post_board
        input_arr = self.board_2_input(board)
        pred = self.network.predict(input_arr)
    
        # Add final state and score to memory
        self.memory.append(input_arr)
        
        # Train network via Q-learning
        real_util = score - 0.5

        for mem_input in self.memory[::-1]:

            # Train network on board and memorized utility
            self.network.train(mem_input, real_util + 0.5)
            
            # Apply discount factor
            real_util = real_util * self.discount
        
        # Clear memory
        self.memory = list()


        
######################
# MONTE CARLO PLAYER #
######################

class MonteCarloPlayer(Player):

    def __init__(self, value, network, nr_samples):
        self.value = value
        self.network = network
        self.nr_samples = nr_samples  
        
    # Requests a move from the player, given a board
    def get_move(self, board, legal_moves):
        
        # Variables that remember best move data
        best_move = -1
        best_pred = 0
        best_input = None
        
        for move in legal_moves:
            
            # Process move on copy of board
            post_board = board.copy()
            played = sum([1 for e in post_board[:, move] if e != 0])
            post_board[board.shape[0]-1-played, move] = self.value
            
            # Get prediction on post_board
            input_arr = self.board_2_input(board)
            pred = self.network.predict(input_arr)
            
            # Update best move and score
            if (best_move == -1 or pred > best_pred):
                best_move = move
                best_pred = pred
                best_input = input_arr
        
        # Determine Monte Carlo based score
        
        game = Game(board, self, Player(2))
        sample_score = game.sample_game(post_board, self, self.nr_samples)
        
        # Train network on score and best choice
        self.network.train(best_input, sample_score)
        
        return best_move
        

########################
# GENETIC STATE PLAYER #
########################

class EndStatePlayer(Player):      
        
    def __init__(self, value, network):
        self.value = value
        self.network = network  
        
    # Requests a move from the player, given a board
    def get_move(self, board, legal_moves):
        
        best_move = -1
        best_pred = 0
        
        for move in legal_moves:
            
            # Process move on copy of board
            post_board = board.copy()
            played = sum([1 for e in post_board[:, move] if e != 0])
            post_board[board.shape[0]-1-played, move] = self.value
            
            # Get prediction on post_board
            input_arr = self.board_2_input(post_board)
            pred = self.network.predict(input_arr)
            
            if best_move == -1 or pred > best_pred:
                best_move = move
                best_pred = pred
        
        return best_move
    
    
    # Function to start the training process    
    def tell_outcome (self, board, score):
    
        # Convert board to input shape
        input_arr = self.board_2_input(board)
        
        # Train network on board and score
        self.network.train(input_arr, score)
 
