import numpy as np
import os

import matplotlib.pyplot as plt
from IPython import display

import lasagne
import lasagne.layers as L
import theano
from theano import tensor as T

class Network():

    def __init__(self, name, input_size, learning_rate):
        
        self.name = name
        self.learning_rate = learning_rate
        
        # Input tensors for data and targets
        self.input_tensor  = T.ftensor4('input')
        self.target_tensor = T.fscalar('targets')

        # Build the network
        self.network = self.build_network(input_size)
        self.train_fn = self.training_function(self.network, self.input_tensor, self.target_tensor, learning_rate)
        self.predict_fn = self.evaluate_function(self.network, self.input_tensor)

    # Construct the network                                           
    def build_network(self, input_size):

        # Input layer
        network = L.InputLayer(shape=(None, input_size[0], input_size[1], input_size[2]), input_var=self.input_tensor)
    
        # Hidden convolutional layers
        network = L.Conv2DLayer(network, 16, 3, W=lasagne.init.HeUniform(np.sqrt(2)), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify)
        network = L.Conv2DLayer(network, 32, 3, W=lasagne.init.HeUniform(np.sqrt(2)), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify)

        # Fully connected layer
        network = L.DenseLayer(network, num_units=100, W=lasagne.init.HeUniform(np.sqrt(2)), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify)
        network = L.DropoutLayer(network, p=0.2)
    
        # Output layer
        network = L.DenseLayer(network, num_units=1, W=lasagne.init.HeUniform(np.sqrt(2)), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.sigmoid)
    
        # Return the built network
        return network

    # Predict evaluation score from board    
    def predict(self, board):
        prediction = self.predict_fn(board)
        prediction += np.random.normal(scale=0.01)
        return prediction

    # Train network on board and label (which can be loss, draw or win)
    def train(self, board, label):
        loss = self.train_fn(board, label)
        params = lasagne.layers.get_all_param_values(self.network)
        
        return loss #(loss, params)

    # Lasagne training function
    def training_function(self, network, input_tensor, target_tensor, learning_rate):
        
        # Get the network output and calculate loss.
        network_output = L.get_output(network)

        loss = lasagne.objectives.squared_error(network_output, target_tensor)
        loss = loss.mean()

        # Get the network parameters and the update function.
        network_params = L.get_all_params(network, trainable=True)
        weight_updates = lasagne.updates.sgd(loss, network_params, learning_rate=learning_rate)
    
        # Construct the training function.
        return theano.function([input_tensor, target_tensor], [loss], updates=weight_updates)

    # Lassagne evaluation function
    def evaluate_function(self, network, input_tensor):
    
        # Get the network output and calculate metrics.
        network_output = L.get_output(network, input_tensor, deterministic=True)
    
        # Construct the evaluation function.
        return theano.function([input_tensor], network_output)

    # Save network
    def save_network(self, winrate, epoch):

        params = L.get_all_param_values(self.network)
        np.savez(os.path.join('../Networks/', self.name + '_' + str(self.learning_rate) + '_' + str(winrate) + '_' + str(epoch) + '.npz'), params=params)
        
    # Load existing network
    def load_network(self, saved_name):

        npz = np.load('../Networks/' + saved_name + '.npz') # load stored parameters
        L.set_all_param_values(self.network, npz['params']) # set parameters

    # Save params
    def get_params(self):

        # Load current params and return them
        return L.get_all_param_values(self.network)

    # Load params
    def set_params(self, params):

        # Set given params in current network
        L.set_all_param_values(self.network, params)
        
