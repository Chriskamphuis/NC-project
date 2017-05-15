from game import *
from player import *

import numpy as np

board = np.zeros((6, 7), dtype=np.int8)

g = Game(board)
p1 = EndStatePlayer(g, 1)
p2 = Player(g, 2)

epochs = 1
for i in range(epochs):   
    winner = g.play_game(p1)
    
    print g.board
    if (winner != 0):
        print "\nPlayer {0} won!".format(winner)
        
    else:
        print "\nThe game was a draw. (Boring!)"
        
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
    
print "Done!"
