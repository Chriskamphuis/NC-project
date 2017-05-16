import numpy as np

from nc17nn.layers import *
from nc17nn.fast_layers import *
from nc17nn.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
      
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.
        
        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        C, H, W = input_dim 
        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        self.params['W1'] = np.random.normal(scale=weight_scale,
                                             size=(num_filters, C, 
                                                   filter_size, filter_size))
        self.params['b1'] = np.zeros(shape=(num_filters))

        self.params['W2'] = np.random.normal(scale=weight_scale, 
                                             size=(num_filters * H * W / 4,
                                                   hidden_dim))
        self.params['b2'] = np.zeros(shape=(hidden_dim))
        
        self.params['W3'] = np.random.normal(scale=weight_scale, 
                                             size=(hidden_dim, num_classes))
        self.params['b3'] = np.zeros(shape=(num_classes))




        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)
        

    def train2(self, board, label, prediction):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        scores = prediction
        l, dx = sigmoid_loss(scores, label)

        # TODO 

        dx, dW3, db3 = affine_backward(dx, cache6)
        dx = relu_backward(dx, cache5)
        dx, dW2, db2 = affine_backward(dx, cache4)
        dx = max_pool_backward_fast(dx, cache3)
        dx = relu_backward(dx, cache2)
        dx, dW1, db1 = conv_backward_fast(dx, cache1)

        weights = [W1, W2, W3]
        
        dW1 = dW1 + W1 * self.reg
        dW2 = dW2 + W2 * self.reg
     
        l2 = self.reg * .5 * np.sum([np.sum(w**2) for w in weights])
        
        loss = l + l2
        grads = {}
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2
        grads['W3'] = dW3
        grads['b3'] = db3

        return loss, grads
