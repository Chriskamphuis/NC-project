from game import *
from player import *
from network import *
from tqdm import tqdm

import numpy as np
import time

'''
    DEFINE PLAYERS AND NETWORKS
'''

net11 = Network("Endstate1", [3, 6, 7], learning_rate=0.01)
#net12 = Network("MonteCarlo1", [3, 6, 7], learning_rate=0.01)
#net13 = Network("Qlearning1", [3, 6, 7], learning_rate=0.1)

#p1 = Player(1, win_in_one = True)
p1 = EndStatePlayer(1, net11, explore_rate=0.1, win_in_one = True)
#p1 = MonteCarloPlayer(1, net12, nr_samples=100, win_in_one = True)
#p1 = QLearningPlayer(1, net13, explore_rate=0.1, discount=0.9, win_in_one = True)

net21 = Network("Endstate2", [3, 6, 7], learning_rate=0.01)
#net22 = Network("MonteCarlo2", [3, 6, 7], learning_rate=0.01)
#net23 = Network("Qlearning2", [3, 6, 7], learning_rate=0.1)

#p2 = Player(2, win_in_one = True)
p2 = EndStatePlayer(2, net21, explore_rate=0.1, win_in_one = True)
#p2 = MonteCarloPlayer(2, net22, nr_samples=100, win_in_one = True)
#p2 = QLearningPlayer(2, net23, explore_rate=0.1, discount=0.9, win_in_one = True)

g = Game(p1, p2)


'''
    MAIN TRAINING LOOP
'''

start = time.time()

epochs = 200
tra_iterations = 1000 #1000
val_iterations = 1000
best_winrate = 0
best_epoch = 0
best_params = None

for i in range(epochs):

    # Parameters to evaluate both training and validation
    wins_p1 = 0.0
    wins_p2 = 0.0
    draws = 0.0
    avg_moves_train = 0.0
    avg_moves_val = 0.0

    # Adjust exploration chance
    explore_rate = max(0.1, 1.0-(0.1*i))
    p1.explore_rate=explore_rate
    p2.explore_rate=explore_rate

    # Training cycle
    for _ in tqdm(range(tra_iterations)):

        (winner, moves) = g.play_game(True)
        avg_moves_train += moves
    
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
    test_game = Game(p1, Player(2))#, board) #Game(p1, Player(2), board)
    
    avg_moves_val = 0.0
    wins_p1 = 0.0
    wins_p2 = 0.0
    draws = 0.0

    for j in tqdm(range(val_iterations)):
        (winner, moves) = test_game.play_game(False)
        avg_moves_val += moves
        
        if (winner == p1.value):
            wins_p1 += 1.0

        if (winner == p2.value):
            wins_p2 += 1.0
            
        elif (winner == 0):
            draws += 1.0
        
        if j < 5:
            test_game.print_board(winner)
        test_game.reset_board()
        test_game.switch_players()
            
    print "Epoch {0}:".format(i)
    print "Average moves/game during training: {0}".format(avg_moves_train/tra_iterations)
    print "Average moves/game during validation: {0}".format(avg_moves_val/val_iterations)
    print "Win percentage P1: {0}".format(wins_p1/(val_iterations))
    print "Win percentage P2: {0}".format(wins_p2/(val_iterations))
    print "Draw percentage: {0}".format(draws/(val_iterations))

    if(wins_p1/(val_iterations) >= best_winrate):
        best_winrate = wins_p1/(val_iterations)
        best_epoch = i
        best_params = p1.get_params()
    
print "Done! Best winrate is {0}% in epoch {1}".format(best_winrate, best_epoch)
print str(time.time() - start) + " seconds"

print "Saving best network..."
p1.set_params(best_params)
p1.save_network(best_winrate, best_epoch)

