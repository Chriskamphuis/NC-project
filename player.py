import game
from random import *

from network import *
import gen_network

import numpy as np
from itertools import permutations
from tqdm import tqdm

import lasagne.layers as L

#################
# RANDOM PLAYER #
#################

class Player():

    def __init__(self, value):
        self.value = value
        self.explore_rate = 0.0
        self.network = None

    # Translates a board into a three-dimensional input array for the neural nets
    # Here dimension 1 is always empty, 2 own moves, 3 opponents' moves
    def board_2_input (self, board):
        shape = board.shape

        # Create input shape
        input_arr = np.zeros((1, 3, shape[0], shape[1]), dtype=np.float32)

        # Fill each dimension
        input_arr[:,0,:,:] = (board == 0).astype(np.float32)
        input_arr[:,1,:,:] = (board == self.value).astype(np.float32)
        input_arr[:,2,:,:] = (input_arr[:,0,:,:] == input_arr[:,1,:,:]).astype(np.float32)

        return input_arr


    # Get a move (standard random move)
    def get_move(self, game, legal_moves, training):

        move = self.check_win_in_one(game, legal_moves)
        if (move != -1):
            return move

        choice = randint(0, len(legal_moves)-1)
        return legal_moves[choice]

    # Check if there is a win in 1 move and returns it
    # Returns -1 if there is no such move
    def check_win_in_one(self, game, legal_moves):    
        # Check if any move is winning
        board = game.board.copy()
        for move in legal_moves:
            winner = game.play_move(move, self.value)
            game.board = board.copy()

            # If it is, play it            
            if (winner == self.value):
                return move
        return -1

    # Function to start the training process (not applicable for random player)
    def tell_outcome(self, board, score):
        return ""

    # Save network
    def save_network(self, winrate, epoch):
        
        self.network.save_network(winrate, epoch)

    # Load network from existing file
    def load_network(self, saved_name):

        self.network.load_network(saved_name)

    # Save params from current network
    def get_params(self):

        return self.network.get_params()

    # Load params in current network
    def set_params(self, params):

        self.network.set_params(params)

        
####################
# END STATE PLAYER #
####################

class EndStatePlayer(Player):

    def __init__(self, value, network, explore_rate=0.1, win_in_one=True):
        self.value = value
        self.network = network
        self.explore_rate = explore_rate
        self.win_in_one = win_in_one

    # Requests a move from the player, given a board
    def get_move(self, game, legal_moves, training):

        # Check for win-in-one if allowed
        if (self.win_in_one):
            move = self.check_win_in_one(game, legal_moves)
            if (move != -1):
                return move

        board = game.board.copy()

        # Choose between exploitation or exploration
        policy_param = random() 

        # If should explore, return random move
        if (training and policy_param <= self.explore_rate):
            choice = randint(0, len(legal_moves)-1)
            return legal_moves[choice]

        # Else exploit as usual
        else:
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

    def __init__(self, value, network, explore_rate=0.1, discount=0.9, win_in_one=True):
        self.value = value
        self.network = network
        self.explore_rate = explore_rate
        self.discount = discount
        self.memory = list()
        self.win_in_one = win_in_one

    # Requests a move from the player, given a board
    def get_move(self, game, legal_moves, training):

        board = game.board.copy()
        
        # Check for win-in-one if allowed
        if (self.win_in_one):
            move = self.check_win_in_one(game, legal_moves)
            if (move != -1):
                
                # Process move on copy of board
                post_board = board.copy()
                played = sum([1 for e in post_board[:, move] if e != 0])
                post_board[board.shape[0]-1-played, move] = self.value

                # Get prediction on post_board
                input_arr = self.board_2_input(post_board)

                # Add board to memory
                self.memory.append(input_arr)
                
                return move

        # Choose between exploitation or exploration
        policy_param = random() 

        # If should explore, return random move
        if (training and policy_param <= self.explore_rate):

            choice = randint(0, len(legal_moves)-1)
            move = legal_moves[choice]

            # Process move on copy of board
            post_board = board.copy()
            played = sum([1 for e in post_board[:, move] if e != 0])
            post_board[board.shape[0]-1-played, move] = self.value

            # Get prediction on post_board
            input_arr = self.board_2_input(post_board)

            # Add board to memory
            self.memory.append(input_arr)
            
            return move

        # Else exploit as usual
        else:
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
                input_arr = self.board_2_input(post_board)
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

    def __init__(self, value, network, nr_samples=10, win_in_one=True):
        self.value = value
        self.network = network
        self.nr_samples = nr_samples
        self.win_in_one = win_in_one

        self.explore_rate = 0.0

    # Requests a move from the player, given a board
    def get_move(self, main_game, legal_moves, training):

        # Check for win-in-one if allowed
        if (self.win_in_one):
            move = self.check_win_in_one(main_game, legal_moves)
            if (move != -1):
                return move

        board = main_game.board.copy()

        # Variables that remember best move data
        best_move = -1
        best_pred = 0
        best_input = None
        best_board = None

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
                best_board = post_board

        # Train network on score and best choice
        if (training):
            
            # Determine Monte Carlo based score
            g = game.Game(Player(1), Player(2), board)
            sample_score = g.sample_game(best_board, self, self.nr_samples)
            self.network.train(best_input, sample_score)

        return best_move


########################
# GENETIC STATE PLAYER #
########################

class GeneticPlayer(Player):

    # Initializes the player.
    def __init__(self, value=1, network=None, population=None):
        self.value = value
        self.network = network
        self.population = population
        if self.population is None:
            self.population = gen_network.Population()

        self.explore_rate = 0.0

    # Requests a move from the player, given a board.
    def get_move(self, board, legal_moves, training):
        if self.network is None:
            raise ValueError('The player does not know a network yet.')
        best_move = -1
        best_pred = 0

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

        return best_move

    '''
    Evaluates the fitness of every network when they play against eachother
    iterations is the amount of games they play as the first player against
    every other network.
    '''
    def evolve(self, iterations):
        fitness = [0 for _ in range(len(self.population.networks))]
        board = np.zeros((6, 7), dtype=np.int8)
        for perm in tqdm(permutations(self.population.networks, 2)):
            p1 = GeneticPlayer(1, network=perm[0])
            p2 = GeneticPlayer(2, network=perm[1])
            for i in range(iterations):
                g = game.Game(p1, p2, board)
                winner = g.play_game(training=True)
                g.reset_board()
                if winner < 2:
                    fitness[self.population.networks.index(perm[winner])] += 1
                else:
                    fitness[self.population.networks.index(perm[0])] += .5
                    fitness[self.population.networks.index(perm[1])] += .5
        self.population.train(fitness)
        self.population.apply_mutation()

    def give_outcome(self):
        pass

