from builtins import object
import numpy as np
import math
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


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
        self.w1 = np.random.random_sample(size=(3,32,32)) * weight_scale
        self.w2 = np.random.random_sample(size=(32, 26, 26)) * weight_scale
        self.w3 = np.random.random_sample(size=(32, 20, 20)) * weight_scale
        self.params = {
            'W1': self.w1,
            'W2': self.w2,
            'W3': self.w3,
            'b1': 0,
            'b2': 0,
            'b3': 0
        }
        self.reg = reg
        self.dtype = dtype


        ############################################################################
        # Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        cache = []
        tmpL, tmpCache = conv_relu_forward(X,W1,b1,conv_param=conv_param)
        cache.append(tmpCache)
        tmpL, tmpCache = max_pool_forward_naive(tmpL,pool_param)
        cache.append(tmpCache)
        tmpL,tmpCache = affine_relu_forward(tmpL,W2,b2)
        cache.append(tmpCache)
        tmpL,tmpCache = affine_forward(tmpL,W3,b3)
        cache.append(tmpCache)


        ############################################################################
        #  Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            scores = tmpL
            return scores

        dw1 = np.zeros(shape=(3, 32, 32))
        dw2 = np.zeros(shape =(32, 26, 26))
        dw3 = np.zeros(shape=(32, 20, 20))
        loss, grads = 0, {
            'W1': dw1,
            'W2': dw2,
            'W3': dw3,
            'b1': 0,
            'b2': 0,
            'b3': 0
        }

        loss,dx = softmax_loss(tmpL,y)
        loss += np.sum(self.params['W1'] ** 2) * self.reg \
                + np.sum(self.params['W2'] ** 2) * self.reg \
                + np.sum(self.params['W3'] ** 2) * self.reg

        dx, dw3, db3 = affine_backward(dx,cache[-1])
        dw3 += self.params['W3'] * self.reg
        grads['W3'] = dw3
        grads['b3'] = db3

        dx, dw2, db2 = affine_relu_backward(dx,cache[-2])
        dw2 += self.params['W2'] * self.reg
        grads['W2'] = dw2
        grads['b2'] = db2

        dx = max_pool_backward_naive(dx,cache[-3])

        dx,dw1,db1 = affine_backward(dx,cache[-4])
        dw1 += self.params['W1'] * self.reg
        grads['W1'] = dw1
        grads['b1'] = db1

        ############################################################################
        #  Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
