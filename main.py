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

board = np.zeros((6, 7), dtype=np.int8)

g = Game(board)
net = Network("Endstate", [3, 6, 7], 0.15)

#p1 = Player(g, 1)
p1 = EndStatePlayer(g, 1, net)
#p1 = MonteCarloPlayer(g, 1, net, 10)
#p1 = QLearningPlayer(g, 1, net, 0.9)

#p2 = Player(g, 2)
p2 = EndStatePlayer(g, 2, net)
#p2 = MonteCarloPlayer(g, 2, net, 10)
#p2 = QLearningPlayer(g, 2, net, 0.9)

start = time.time()

epochs = 1000
winners = 0.0
for i in tqdm(range(epochs)):   
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
    
    winners += winner
    
print "Done!"
print str(time.time() - start) + " seconds"

print (winners)/epochs
