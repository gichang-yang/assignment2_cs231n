from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {
            #'W1':np.random.normal(scale=weight_scale, size=(input_dim,hidden_dim)),
            'W1': np.random.randn(input_dim, hidden_dim) * weight_scale,
            #'W2':np.random.normal(scale=weight_scale, size=(hidden_dim,num_classes)),
            'W2': np.random.randn(hidden_dim, num_classes) * weight_scale,
            'b1':0,
            'b2':0
        }
        self.reg = reg



    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None

        L1,l1Cache = affine_relu_forward(X,self.params['W1'], self.params['b1'])
        affined, l2cache = affine_forward(L1, self.params['W2'],self.params['b2'])

        ############################################################################
        # Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            scores = affined
            return scores

        loss, grads = 0, {
            'W1':None,
            'W2':None,
            'b1':None,
            'b2':None,
        }
        l, dx = softmax_loss(affined, y)
        dl2,grads['W2'],grads['b2'] = affine_backward(dx,l2cache)
        dl1,grads['W1'],grads['b1'] = affine_relu_backward(dl2,l1Cache)

        loss = l + np.sum(self.params['W1'] ** 2) * self.reg * 0.5\
               + np.sum(self.params['W2'] ** 2) * self.reg * 0.5
        ############################################################################
        #  Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {
            'W1':np.full(shape=(input_dim,hidden_dims[0]),fill_value=np.random.random_integers(0,1)) * weight_scale,
            'b1':np.zeros(shape=hidden_dims[0],),
            'gamma1':np.ones(shape=(hidden_dims[0],)),
            'beta1':np.zeros(shape=(hidden_dims[0],)),
        }

        for i in range(1,len(hidden_dims)):
            self.params.setdefault('W'+str(i+1),np.random.randn(hidden_dims[i-1],hidden_dims[i]) * weight_scale)
            self.params.setdefault('b'+str(i+1),np.random.randn(hidden_dims[i],) * weight_scale)
            self.params.setdefault('gamma'+str(i+1),np.ones(shape=(hidden_dims[i],)))
            self.params.setdefault('beta'+str(i+1),np.zeros(shape=(hidden_dims[i],)))
        self.params.setdefault('W' + str(self.num_layers),
                               np.random.randn(
                                   hidden_dims[len(hidden_dims)-1],num_classes
                                   ) * weight_scale
                               )
        self.params.setdefault('b' + str(self.num_layers), np.zeros(shape=(num_classes,)))
        #print(self.params)
        ############################################################################
        # Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {'mode':'test','p':dropout}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{
                'mode': 'train',
                'eps' : 1e-5,
                'momentum': 0.9
            } for i in range(self.num_layers - 1)]
            print(self.bn_params)
        else:
            self.bn_params = [{
                'mode': 'test',
            } for i in range(self.num_layers - 1)]
        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        cache = []
        affined = X

        print('pn_params',self.bn_params)
        for i in range(self.num_layers-1):
            print('W',str(i+1))
            affined, tmp = affine_forward(affined, self.params['W'+str(i+1)], self.params['b' + str(i + 1)])
            cache.append(tmp)
            affined, tmp = batchnorm_forward(
                affined, gamma=self.params['gamma'+str(i+1)],
                beta=self.params['beta'+str(i+1)],
                bn_param=self.bn_params[i]
            )
            cache.append(tmp)
            affined,tmp = relu_forward(affined)
            cache.append(tmp)
            affined,tmp = dropout_forward(affined,dropout_param=self.dropout_param)
            cache.append(tmp)
        affined,tmp = affine_forward(affined,self.params['W'+str(self.num_layers)],self.params['b'+str(self.num_layers)])
        cache.append(tmp)


        scores =  affined
        ############################################################################
        #  Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        loss, dx = softmax_loss(affined, y)
        tmp_gradX,tmp_gradW,tmp_gradB = affine_backward(dx,cache[-1])
        grads['W'+str(self.num_layers)] = tmp_gradW
        grads['b'+str(self.num_layers)] = tmp_gradB

        for i in range(1,self.num_layers):
            tmp_gradX= dropout_backward(tmp_gradX, cache[2 - 4*i])
            tmp_gradX = relu_backward(tmp_gradX, cache[1 - 4 * i])
            tmp_gradX,tmp_gradGamma,tmp_gradBeta = batchnorm_backward(tmp_gradX,cache[0 - 4 * i])
            tmp_gradX,tmp_gradW,tmp_gradB = affine_backward(tmp_gradX,cache[-(1 + 4*i)])
            grads['W' + str(self.num_layers-i)] = tmp_gradW
            grads['b' + str(self.num_layers-i)] = tmp_gradB
            grads['gamma'+str(self.num_layers-i)] = tmp_gradGamma
            grads['beta'+str(self.num_layers-i)] = tmp_gradBeta

        loss = tmp_gradX
        ############################################################################
        # Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
