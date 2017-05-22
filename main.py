from game import *
from player import *
from network import *
from tqdm import tqdm

import numpy as np
import time

##########
def printGame(board, winner):
    print board
    if (winner != 0):
        print "\nPlayer {0} won!".format(winner)
        
    else:
        print "\nThe game was a draw. (Boring!)"
#########

net = Network("Endstate1", [3, 6, 7], 0.15)

#p1 = Player(1)
p1 = EndStatePlayer(1, net)
#p1 = MonteCarloPlayer(1, net1, 10)
#p1 = QLearningPlayer(1, net1, 0.9)

net2 = Network("Endstate2", [3, 6, 7], 0.15)

#p2 = Player(2)
p2 = EndStatePlayer(2, net)
#p2 = MonteCarloPlayer(2, net2, 10)
#p2 = QLearningPlayer(2, net2, 0.9)

board = np.zeros((6, 7), dtype=np.int8)
g = Game(board, p1, p2)

start = time.time()

epochs = 25
iterations = 100

winners = 0.0
for i in tqdm(range(epochs)):   

    # Training cycle
    for _ in range(iterations):

        winner = g.play_game()
    
        #printGame(g.board, winner)
        
        if (winner == 0):
            p1.tell_outcome(g.board, 0.5)
            p2.tell_outcome(g.board, 0.5)
        elif (winner == p1.value):
            p1.tell_outcome(g.board, 1.0)
            p2.tell_outcome(g.board, 0.0)
        else:
            p1.tell_outcome(g.board, 0.0)
            p2.tell_outcome(g.board, 1.0)
    
        g.reset_board()
        g.switch_players()
        
    # Test cycle
    wins = 0.0
    draws = 0.0
    test_game = Game(board, p1, Player(2))
    for _ in range(iterations):
        winner = test_game.play_game()
        
        if (winner == p1.value):
            wins += 1.0
            
        elif (winner != 0):
            draws += 1.0
            
    print "Epoch {0}:".format(i)
    print "Win percentage: {0}".format(wins/iterations)
    print "Draw percentage: {0}".format(draws/iterations)

    
print "Done!"
print str(time.time() - start) + " seconds"

