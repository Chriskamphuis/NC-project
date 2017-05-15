from game import *
from player import *

g = Game()
p1 = EndStatePlayer(g, 1)
p2 = Player(g, 2)

epochs = 1000
for i in range(epochs):   
    winner = g.play_game(p1)
    
    #if (winner != 0):
    #    print "\nPlayer {0} won!".format(winner)
        
    #else:
    #    print "The game was a draw. (Boring!)"
    
    g.reset_board()
    g.switch_players()
    
print "Done!"
