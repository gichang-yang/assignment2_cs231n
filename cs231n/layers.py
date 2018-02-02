from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    x_flatten=np.reshape(x,newshape=[x.shape[0],-1])
    out = np.matmul(x_flatten,w) + b
    ###########################################################################
    # Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    dx = np.matmul(dout,np.transpose(w)).reshape(x.shape)
    dw = np.matmul(np.transpose(np.reshape(x,(x.shpe[0],-1))),dout)
    db = b
    ###########################################################################
    # Implement the affine backward pass.                               #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None

    out = x + np.fabs(x) * 0.5
    ###########################################################################
    # Implement the ReLU forward pass.                                  #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    dx= (dout > 0) * 1
    ###########################################################################
    # Implement the ReLU backward pass.                                 #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        mu = 1./N * np.sum(x,axis=0)

        st_mean = -1 * (mu - x)
        running_mean = st_mean

        exp_mean = st_mean ** 2

        st_var = 1./N * np.sum(exp_mean,axis=0)
        running_var = st_var

        sq = np.sqrt(st_var + eps)
        reversed_sq = 1. / sq
        normed = st_mean * reversed_sq
        out = gamma * normed + beta
        cache = (out, gamma, normed, reversed_sq, sq, st_var, st_mean, x)


        #######################################################################
        #      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        mu = 1. / N * np.sum(x, axis=0)

        st_mean = -1 * (mu - x)
        running_mean = momentum*running_mean - (1-momentum) * st_mean

        exp_mean = running_mean ** 2

        st_var = 1. / N * np.sum(exp_mean, axis=0)
        running_var = running_var*momentum - (1-momentum) * st_var

        sq = np.sqrt(running_var + eps)
        reversed_sq = 1. / sq
        normed = running_mean * reversed_sq
        out = gamma * normed + beta
        cache = (out, gamma, normed, reversed_sq, sq, running_var, running_mean, x)
        #######################################################################
        # Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    N,D = dout.shape
    out, gamma, normed, reversed_sq, sq, var, mean, x = cache

    dnormed = gamma * dout
    dreversed_sq = mean * dnormed
    dsq = -1 * (sq ** -2) * dreversed_sq
    dvar = 0.5 * (sq ** -0.5) * dsq
    dexp_mean = 1./N * np.ones(shape=dout.shape) * dvar
    dmean = 2 * mean * dexp_mean + reversed_sq * dout
    dmu =  -1 * dmean
    dx = dmean + dmu * 1./N * np.ones(shape=dout.shape)

    dgamma = 1./N * np.sum(normed * dout,axis=0)
    dbeta = 1./N * np.sum(dout, axis=0)


    ###########################################################################
    # Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """

    dx, dgamma, dbeta = None, None, None
    out, gamma, normed, reversed_sq, sq, var, mean, x = cache
    N, D = x.shape
    dhat = dout * gamma
    dvar = np.sum((x - mean)*(-0.5)*(sq**-3) *dhat, axis=0)
    dmean =1./N * np.sum( dvar * (-2 * (x - mean)) , axis=0) - (sq ** -1) * dhat
    dx = dmean * 1./N * + np.sum(2*(x-dmean)*dvar,axis=0) + dvar * 2./N * (x-mean) + dhat * 1./sq

    dgamma = np.mean(dout * normed,axis=0)
    dbeta = np.mean(dout,axis=0)


    ###########################################################################
    # Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = np.ones_like(x)
    out = None

    if mode == 'train':
        shape = x.shape
        num = np.sum(x.shape)
        x_reshaped = np.reshape(x,newshape=[int(num),])
        choosen_list = np.random.choice(range(num),int(num * p))
        for i in choosen_list:
            x_reshaped[i] = 0

        out = np.reshape(x_reshaped,newshape=shape)
        mask = np.putmask(mask, x_reshaped==0, 0)

        #######################################################################
        #  Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        out = x
        #######################################################################
        # Implement the test phase forward pass for inverted dropout.   #
        #######################################################################

        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        dx = dout * mask
        #######################################################################
        #  Implement training phase backward pass for inverted dropout   #
        #######################################################################
        pass
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    stride = conv_param['stride']
    pad = conv_param['pad']

    H_prime = 1 + (H + 2 * pad - HH) / stride
    W_prime = H_prime
    out = np.zeros(shape=(N,F,H_prime,W_prime))

    padded_x = np.zeros(shape=(N,C,x.shape[2]+pad*2 , x.shape[3]+pad*2))
    padded_x[:,:,pad:pad+x.shape[2],pad:pad+x.shape[3]] = x


    for i in range(H_prime):
        for n in range(W_prime):
            for m in range(N):
                out_mat = w[:,:,:,:]*padded_x[m,:,stride*i:i*stride+HH,stride*n:stride*n+WW] + b
                out[m,:,i,n] = np.sum(out_mat,axis=(0,2,3))

    ###########################################################################
    # Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    x, w, b, conv_param = cache
    pad = conv_param['pad']
    stride = conv_param['stride']


    N, C, H, W = x.shape
    dw = np.zeros_like(W)
    F, C, HH, WW = w.shape

    H_prime = 1 + (H + 2 * pad - HH) / stride
    W_prime = H_prime



    d_padded_x = np.zeros(shape=(N, C, x.shape[2] + pad * 2, x.shape[3] + pad * 2))

    for i in range(H_prime):
        for n in range(W_prime):
            for m in range(N):
                rev = dout[m,:,i,n]

                d_padded_x[m, :, stride * i:i * stride + HH, stride * n:stride * n + WW] += np.sum(w * rev, axis=0)
                dw += x * rev
                db += np.sum(np.ones_like(W) * rev,axis=(1,2,3))

    dx = d_padded_x[:,:,pad:pad+x.shape[2],pad:pad+x.shape[3]]
    ###########################################################################
    # Implement the convolutional backward pass.                        #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None

    N, C, H, W = x.shape


    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    H_prime = 1 + (H + 2 *  - pool_height) / stride
    W_prime = 1 + (W + 2 *  - pool_width) / stride

    mask_x = np.zeros_like(x)
    result = np.zeros(shape=(N,C,H_prime,W_prime))
    for i in range(H_prime):
        for n in range(W_prime):
            for m in range(N):
                result[m,:,i,n] = np.max(x[m,:,i*stride:i*stride + pool_height, i*stride:i*stride+pool_width],axis=(1,2))

    out = result

    ################################################
    # Implement the max pooling forward pass                            #
    ###########################################################################

    ##########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    x,pool_param = cache
    N, C, H, W = x.shape


    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    H_prime = 1 + (H + 2 * - pool_height) / stride
    W_prime = 1 + (W + 2 * - pool_width) / stride


    dx = np.zeros_like(x)

    for i in range(H_prime):
        for n in range(W_prime):
            for m in range(N):
                mask = x[m,:,i*stride:i*stride + pool_height, i*stride:i*stride+pool_width]
                tmp_x = np.transpose((((
                    np.transpose(x[m, :, i * stride:i * stride + pool_height, i * stride:i * stride + pool_width])
                    -
                    np.transpose(np.max(x[m,:,i*stride:i*stride + pool_height, i*stride:i*stride+pool_width],axis=(1,2)))
                    )== 0) + 0) * np.transpose(dout[m,:,i,n])
                )
                dx[m,:,i * stride:i * stride + pool_height, i * stride:i * stride + pool_width] += tmp_x

    ###########################################################################
    #  Implement the max pooling backward pass                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.
    이전에 구현하였던 batch norm은 2차원 (N,D) 에만 국한된 구현이었습니다
    하지만 우리가 앞으로 이용할 CNN에서는 convnet, maxpooling 등등을 거치다보면 2차원으로는
    구현에 한계가 있습니다
    따라서 여러 고차원의 input에도 대응할 수 있게 구현하시기 바랍니다.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    N, C, H, W = x.shape
    tmp_out, cache = batchnorm_forward(np.transpose(x,(0,3,2,1)).reshape(shape=(N*H*W,C)),gamma,beta,bn_param)

    out = tmp_out.transpose((0,3,2,1)).reshape(shape=(N,C,H,W))

    ###########################################################################
    # Implement the forward pass for spatial batch normalization.             #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None
    N, C, H, W = dout.shape
    tmp_dx,dgamma,dbeta =  batchnorm_backward(np.transpose(dout, (0, 3, 2, 1)).reshape(shape=(N * H * W, C)),cache)
    dx = tmp_dx.transpose((0, 3, 2, 1)).reshape(shape=(N, C, H, W))

    ###########################################################################
    # Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
