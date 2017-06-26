### NC-project

This repository contains the code for the game Connect Four with different convolutional evaluation network agents. 
The code was developed for a project for the course Natural Computing at the Radboud University Nijmegen.

The following files contain the important information for the project:

 * game.py : Contains the class with the most important information about the board states. 
 * player.py : Contains a base player class and a player class for all players.
 * network.py : Contains all the information about the networks for all players except the evolutionary player.
 * gen_network.py : Contains the network for the evolutionary player, also code to train it.
 * trainer.py : Contains the code to train all the player except the evolutionary player.
 * tester.py : Contains the code to evaluate the players.

In order to train players using trainer.py, trainer.py needs to be altered for that specific player. 
Every player and their respective networks have a line that needs to be uncommented in order to
train that specific player. When we trained the players, they were trained against a player 
using the same approach (for example Qlearn vs Qlearn), but feel free to train otherwise.
Player 1 is validated against a random player each epoch. The network of player 1 with the best winrate 
during validation gets saved at the end of training in the folder 'Networks'.

The Evolutionary player is not included in the trainer.py above as it uses a different strategy for 
training by evolving the weights of the network. To train the evolutionary player, gen_player.py needs 
to be run. It is possible to set the parameters at the bottom of the file. This player is trained using 
a pool of networks that  train against each other. We trained our network using elitism, and a crossover 
rate and mutation rate of both 0.01.

The best networks of each approach can be tested aganst each other with the tester.py class.
Tester.py needs to be altered by uncommenting the right players, network set-ups and trained networks. 
The Evolutionary player does not need a general network set-up. Random players can also be used
to test against by uncommeting the Player() classes. The random players both do not use a general 
network set-up or trained network.


