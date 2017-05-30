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

net1 = Network("Endstate1", [3, 6, 7], 0.01)

#p1 = Player(1)
p1 = EndStatePlayer(1, net1, 0.1)
#p1 = MonteCarloPlayer(1, net, 10)
#p1 = QLearningPlayer(1, net1, 0.1, 0.9)

net2 = Network("Endstate2", [3, 6, 7], 0.2)

#p2 = Player(2)
#p2 = EndStatePlayer(2, net2, 0.1)
#p2 = MonteCarloPlayer(2, net, 10)
p2 = QLearningPlayer(2, net2, 0.25, 0.9)

#board = np.zeros((6, 7), dtype=np.int8)
g = Game(p1, p2)#, board)

start = time.time()

epochs = 10 #25
iterations = 1000 #1000

winners = 0.0
for i in range(epochs):   

    # Training cycle
    for _ in tqdm(range(iterations)):

        winner = g.play_game(True)
    
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
        
    # Validation cycle
    wins_p1 = 0.0
    wins_p2 = 0.0
    draws = 0.0
    test_game = Game(p1, p2)#, board) #Game(p1, Player(2), board)
    for _ in tqdm(range(iterations/500)):
        winner = test_game.play_game(False)
        
        if (winner == p1.value):
            wins_p1 += 1.0

        if (winner == p2.value):
            wins_p2 += 1.0
            
        elif (winner == 0):
            draws += 1.0

        test_game.print_board()
        test_game.reset_board()
        test_game.switch_players()
            
    print "Epoch {0}:".format(i+1)
    print "Win percentage P1: {0}".format(wins_p1/(iterations/500))
    print "Win percentage P2: {0}".format(wins_p2/(iterations/500))
    print "Draw percentage: {0}".format(draws/(iterations/500))

    
print "Done!"
print str(time.time() - start) + " seconds"

