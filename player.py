from game import *
from random import randint

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


class EndStatePlayer(Player):
    def get_move(self, legal_moves):
        
        for move in legal_moves:
            utility = forward_pass
        
        choice = randint(0, len(legal_moves)-1)
        
        return legal_moves[choice]
