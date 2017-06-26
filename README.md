### NC-project

This repository contains the code of the game connect-4 for a project for the course natural computing.

The following files contain the important information for the project:

 * game.py : Contains the class with the most important information about the boardstates. 
 * player.py : Contains a base player class and a player class for all players except the evolutionary player.
 * network.py : Contains all the information for the networks all players except the evolutionary player uses.
 * gen_network.py : Contains the network for the evolutionary player.
 * gen_player.py : Contains the class for the evolutionary player, also code to train it.
 * trainer.py : Contains the code to train all the player except the evolutionary player.
 * tester.py : Contains the code to evaluate the players.

In order to train players using train.py, train.py needs to be altered for that specific player. 
Every player and their respective networks have a line that needs to be uncommented in order to
train that specific player. When we trained the players they were trained against a player 
using the same kind of training style. (For example Qlearn vs Qlearn).

In order to train the evolutionary player, gen_player.py needs to be run. It is possible to set
the parameters at the bottom of the file. This player is trained using a pool of networks that 
train against eachother. We trained our network using elitism, and a crossover rate and 
mutation rate of both 0.01.
