import numpy as np

from fast_layers import *
from layer_utils import *


class FiveLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - conv - relu - 2x2 max pool - conv - relu -
  2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=[16,32,64], filter_size=3,
               hidden_dim=100, num_classes=2, weight_scale=1e-3, reg=0.0,
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
                                         size=(num_filters[0], C, 
                                               filter_size, filter_size))
    self.params['b1'] = np.zeros(shape=(num_filters[0]))

    self.params['W2'] = np.random.normal(scale=weight_scale,
                                         size=(num_filters[1], num_filters[0],
                                               filter_size, filter_size))
    self.params['b2'] = np.zeros(shape=(num_filters[1]))

    self.params['W3'] = np.random.normal(scale=weight_scale,
                                         size=(num_filters[2], num_filters[0],
                                               filter_size, filter_size))
    self.params['b3'] = np.zeros(shape=(num_filters[2]))

    self.params['W4'] = np.random.normal(scale=weight_scale, 
                                         size=(num_filters[2] * H * W / 4,
                                               hidden_dim))
    self.params['b4'] = np.zeros(shape=(hidden_dim))
    
    self.params['W5'] = np.random.normal(scale=weight_scale, 
                                         size=(hidden_dim, num_classes))
    self.params['b5'] = np.zeros(shape=(num_classes))




    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    W5, b5 = self.params['W5'], self.params['b5']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    conv1, cache1 = conv_forward_fast(X, W1, b1, conv_param)
    relu1, cache2 = relu_forward(conv1)
    pool1, cache3 = max_pool_forward_fast(relu1, pool_param)

    conv2, cache4 = conv_forward_fast(pool1, W2, b2, conv_param)
    relu2, cache5 = relu_forward(conv2)
    pool2, cache6 = max_pool_forward_fast(relu2, pool_param)

    conv3, cache7 = conv_forward_fast(pool2, W3, b3, conv_param)
    relu3, cache8 = relu_forward(conv3)
    pool3, cache9 = max_pool_forward_fast(relu3, pool_param)
    
    affine1, cache10 = affine_forward(pool3, W4, b4)
    relu4, cache11 = relu_forward(affine1)
    scores, cache12 = affine_forward(relu4, W5, b5)
    
    l, dx = softmax_loss(scores, y) 
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
        
    dx, dW5, db5 = affine_backward(dx, cache12)
    dx = relu_backward(dx, cache11)
    dx, dW4, db4 = affine_backward(dx, cache10)
    dx = max_pool_backward_fast(dx, cache9)
    dx = relu_backward(dx, cache8)
    dx, dW3, db3 = conv_backward_fast(dx, cache7)
    dx = max_pool_backward_fast(dx, cache6)
    dx = relu_backward(dx, cache5)
    dx, dW2, db2 = conv_backward_fast(dx, cache4)
    dx = max_pool_backward_fast(dx, cache3)
    dx = relu_backward(dx, cache2)
    dx, dW1, db1 = conv_backward_fast(dx, cache1)

    weights = [W1, W2, W3, W4, W5]
    
    dW1 = dW1 + W1 * self.reg
    dW2 = dW2 + W2 * self.reg
    dW3 = dW3 + W3 * self.reg
    dW4 = dW4 + W4 * self.reg
 
    l2 = self.reg * .5 * np.sum([np.sum(w**2) for w in weights])
    
    loss = l + l2
    grads = {}
    grads['W1'] = dW1
    grads['b1'] = db1
    grads['W2'] = dW2
    grads['b2'] = db2
    grads['W3'] = dW3
    grads['b3'] = db3
    grads['W4'] = dW4
    grads['b4'] = db4
    grads['W5'] = dW5
    grads['b5'] = db5


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
