from game import *
from player import *
from network import *
from tqdm import tqdm

import numpy as np
import time

'''
    DEFINE PLAYERS AND NETWORKS
'''

net11 = Network("Endstate1", [3, 6, 7], learning_rate=0.2)
#net12 = Network("MonteCarlo1", [3, 6, 7], learning_rate=0.01)
#net13 = Network("Qlearning1", [3, 6, 7], learning_rate=0.1)

#p1 = Player(1)
p1 = EndStatePlayer(1, net11, explore_rate=0.1, win_in_one = True)
#p1 = MonteCarloPlayer(1, net12, nr_samples=10, win_in_one = True)
#p1 = QLearningPlayer(1, net13, explore_rate=0.1, discount=0.9, win_in_one = True)

net21 = Network("Endstate2", [3, 6, 7], learning_rate=0.2)
#net22 = Network("MonteCarlo2", [3, 6, 7], learning_rate=0.01)
#net23 = Network("Qlearning2", [3, 6, 7], learning_rate=0.1)

#p2 = Player(2)
p2 = EndStatePlayer(2, net21, explore_rate=0.1, win_in_one = True)
#p2 = MonteCarloPlayer(2, net22, nr_samples=10, win_in_one = True)
#p2 = QLearningPlayer(2, net23, explore_rate=0.1, discount=0.9, win_in_one = True)

p1.load_network('Endstate1_0.2_0.69_0')
#p2.load_network('Qlearning1_0.1_94_7')

g = Game(p1, p2)


'''
    MAIN TESTING LOOP
'''

start = time.time()

# Parameters to evaluate during testing
test_iterations = 100
wins_p1 = 0.0
wins_p2 = 0.0
draws = 0.0
avg_moves = 0.0
winning_player = 0

for i in range(test_iterations):

    (winner, moves) = g.play_game(False)
    avg_moves += moves
        
    if (winner == p1.value):
        wins_p1 += 1.0

    if (winner == p2.value):
        wins_p2 += 1.0
            
    elif (winner == 0):
        draws += 1.0

    if i < 5:
        g.print_board(winner)

    g.reset_board()
    g.switch_players()

if(wins_p1 >= wins_p2):
    winning_player = p1.value
else:
    winning_player = p2.value
    
print "Done! Player {0} wins with a winrate of {1}%".format(winning_player, wins_p1/test_iterations)
print str(time.time() - start) + " seconds"  
