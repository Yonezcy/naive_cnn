"""Builds the CIFAR-10 network."""
import numpy as np
import sys
import os
import urllib
import tarfile
from datetime import datetime
import time

import cifar10_input


DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz' # Data Download URL
data_dir = '/Users/apple/desktop/cifar-10-batches-py' # Data directory
batch_size = 128 # Batch size


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward propagation for convolutional neural network

    Args:
        x: Input data of shape(M, channels, height, width)
        w: Filter weights of shape(filter_num, channels, filter_height, filter_width)
        b: Biases of shape(filter_num, 1)
        conv_param: A dictionary with the following keys:
            'stride': The number of pixels between adjacent receptive fields in the
                      horizontal and vertical directions
            'pad': The number of pixels that will be used to zero-pad the input

    Returns:
        out: Output data, of shape(N, F, H', W') where H' and W' are given by
        H' = 1 + (H + 2 * pad - HH) / stride
        W' = 1 + (W + 2 * pad - WW) / stride
        cache: (x, w, b, conv_param) Store the variables to compute the derivatives
    """

    # Compute the dimensions
    M, channels, height, width = x.shape
    filter_num, channels, filter_height, filter_width = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']

    # Compute the dimensions of output
    new_H = 1 + int((height + 2 * pad - filter_height) / stride)
    new_W = 1 + int((width + 2 * pad - filter_width) / stride)
    out = np.zeros((M, filter_num, new_H, new_W))

    # Forward prop
    for m in range(M):
        for f in range(filter_num):
            conv_newH_newW = np.ones((new_H, new_W)) * b[f]
            for c in range(channels):
                padded_x = np.lib.pad(x[m, c], pad_width=pad, mode='constant',
                                      constant_values=0)
                for i in range(new_H):
                    for j in range(new_W):
                        conv_newH_newW[i, j] += np.sum(padded_x[i*stride: \
                            i*stride+filter_height, j*stride: j*stride+filter_width]\
                            * w[f,c,:,:])
                        out[m, f] = conv_newH_newW

    cache = (x, w, b, conv_param)

    return out, cache

def conv_backward_naive(dout, cache):
    """A naive implementation of the backward propagation for convolutional neural network

    Args:
        dout: Upstream derivatives
        cache: A tuple of (x, w, b, conv_param) as in forward prop

    Returns:
        dx: Gradients with respect to x
        dw: Gradients with respect to w
        db: Gradients with respect to b
    """

    x, w, b, conv_param = cache
    pad = conv_param['pad']
    stride = conv_param['stride']
    M, channels, height, width = x.shape
    filter_num, channels, filter_height, filter_width = w.shape
    M, filter_num, new_H, new_W = dout.shape

    padded_x = np.lib.pad(x,
                          ((0, 0), (0, 0), (pad, pad), (pad, pad)),
                          mode='constant',
                          constant_values=0)
    padded_dx = np.zeros_like(padded_x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    # Backward prop
    for m in range(M):
        for f in range(filter_num):
            for i in range(new_H):
                for j in range(new_W):
                    db[f] += dout[m, f, i, j]
                    dw[f] += padded_x[m, :, i*stride: i*stride+filter_height,
                             j*stride: j*stride+filter_width] * dout[m, f, i, j]
                    padded_dx[m, :, i*stride: i*stride+filter_height,
                    j*stride: j*stride+filter_width] += w[f] * dout[m, f, i, j]

    dx = padded_dx[:, :, pad:pad+height, pad:pad+width]

    return dx, dw, db

def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward propagation for max pooling layer

    Args:
        x: Input data, of shape(M, channels, height, width)
        pool_param: dictionary with the following keys:
            'pool_height': The height of each pooling region
            'pool_width': The width of each pooling region
            'stride': The distance between adjacent pooling regions

    Returns:
        out: Output data
        cache: (x, pool_param)
    """

    M, channels, height, weight = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    pool_stride = pool_param['stride']
    new_H = 1 + int((height - pool_height) / pool_stride)
    new_W = 1 + int((weight - pool_width) / pool_stride)

    out = np.zeros((M, channels, new_H, new_W))

    # Forward prop
    for m in range(M):
        for c in range(channels):
            for i in range(new_H):
                for j in range(new_W):
                    out[m, c, i, j] = np.max(x[m, c, i*pool_stride: i*pool_stride \
                                            +pool_height, j*pool_stride: \
                                            j*pool_stride+pool_width])

    cache = (x, pool_param)

    return out, cache

def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward propagation for max pooling layer

    Args:
        dout: Upstream derivatives
        cache: A tuple of (x, pool_param) as in the forward pass

    Returns:
        dx: Gradients with respect to x
    """

    x, pool_param = cache
    M, channels, height, width = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    pool_stride = pool_param['stride']
    new_H = 1 + int((height - pool_height) / pool_stride)
    new_W = 1 + int((width - pool_width) / pool_stride)

    dx = np.zeros_like(x)

    # Backward prop
    for m in range(M):
        for c in range(channels):
            for i in range(new_H):
                for j in range(new_W):
                    window = x[m, c, i*pool_stride: i*pool_stride+pool_height,
                             j*pool_stride: j*pool_stride+pool_width]
                    dx[m, c, i*pool_stride: i*pool_stride+pool_height, j*pool_stride\
                    : j*pool_stride+pool_width] = (window == np.max(window)) * \
                    dout[m, c, i, j]

    return dx




def linear_forward(A, W, b):
    """Implement the linear part of a layer's forward propagation.

    Args:
        A: activations from previous layer (or input data): (size of previous layer, number of examples)
        W: weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b: bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
        Z: The input of the activation function, also called pre-activation parameter
        cache : A python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    Z = np.dot(W, A) + b

    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache

def linear_activation_forward(A_prev, W, b, activation, gamma, beta, bn_param):
    """Implement the forward propagation for the LINEAR->ACTIVATION layer

    Args:
        A_prev: activations from previous layer (or input data): (size of previous layer, number of examples)
        W: weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b: bias vector, numpy array of shape (size of the current layer, 1)
        activation: the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
        A: The output of the activation function, also called the post-activation value
        cache: A python dictionary containing "linear_cache" and "activation_cache";
               stored for computing the backward pass efficiently
    """


    # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
    if activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        Z, batchnorm_cache = batchnorm_forward(Z, gamma, beta, bn_param)
        A, activation_cache = relu(Z)
    elif activation == 'softmax':
        Z, linear_cache = linear_forward(A_prev, W, b)
        Z, batchnorm_cache = batchnorm_forward(Z, gamma, beta, bn_param)
        A, activation_cache = softmax(Z)

    # assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache, batchnorm_cache)

    return A, cache

def linear_backward(dZ, cache):
    """Implement the linear portion of backward propagation for a single layer (layer l)

    Args:
        dZ: Gradient of the cost with respect to the linear output (of current layer l)
        cache: tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
        dA_prev: Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW: Gradient of the cost with respect to W (current layer l), same shape as W
        db: Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1. / m * np.dot(dZ, A_prev.T)
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache):
    """Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Args:
        dA: post-activation gradient for current layer l
        cache: tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation: the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
        dA_prev: Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW: Gradient of the cost with respect to W (current layer l), same shape as W
        db: Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache, batchnorm_cache = cache
    dZ = relu_backward(dA, activation_cache)
    dZ, dgamma, dbeta = batchnorm_backward(dZ, batchnorm_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)



    return dA_prev, dW, db, dgamma, dbeta




def batchnorm_forward(x, gamma, beta, bn_param):
    """Batch normalization of activation outputs.

    Args:
        x: activation outputs.
        gamma: parameter of formula (x_norm = gamma * x_norm + beta).
        beta: parameter of formula (x_norm = gamma * x_norm + beta).
        bn_param: A python dictionary of {mode, epsilon, momentum, running_mean, running_var}.

    Returns:
        out: Normalization of activation outputs.
        cache: Tuple of values (x, sample_mean, sample_var, x_hat, eps, gamma, beta) we stored.
    """

    mode = bn_param['mode']
    eps = bn_param['eps']
    momentum = bn_param['momentum']

    M, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)

        out = gamma * x_hat + beta

        cache = (x, sample_mean, sample_var, x_hat, eps, gamma, beta)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

    elif mode == 'test':
        out = (x - running_mean) * gamma / np.sqrt(running_var + eps) + beta

    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache

def batchnorm_backward(dout, cache):
    """Backward propagation of batch normalization.

    Args:
        dout: The derivative with respect to last variable.
        cache: Tuple of values (x, sample_mean, sample_var, x_hat, eps, gamma, beta)
               we stored in forward propagation.

    Returns:
        dx: The derivative with respect to activation output.
        dgamma: The derivative with respect to gamma.
        dbeta: The derivative with respect to beta.
    """

    x, mean, var, x_hat, eps, gamma, beta = cache
    M = x.shape[0]
    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta = np.sum(dout * 1.0, axis=0)
    dx_hat = dout * gamma
    dx_hat_numerator = dx_hat / np.sqrt(var + eps)
    dx_hat_denominator = np.sum(dx_hat * (x-mean), axis=0)
    dx_1 = dx_hat_numerator
    dvar = -0.5 * ((var + eps) ** (-1.5)) * dx_hat_denominator

    dmean = -1.0 * np.sum(dx_hat_numerator, axis=0) + \
            dvar * np.mean(-2.0 * (x-mean), axis=0)

    dx_var = dvar * 2.0 / M * (x-mean)
    dx_mean = dmean * 1.0 / M

    dx = dx_1 + dx_var + dx_mean

    return dx, dgamma, dbeta

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Spatial batch normalization of activation outputs.

    Args:
        x: activation outputs.
        gamma: parameter of formula (x_norm = gamma * x_norm + beta).
        beta: parameter of formula (x_norm = gamma * x_norm + beta).
        bn_param: A python dictionary of {mode, epsilon, momentum, running_mean, running_var}.

    Returns:
        out: Normalization of activation outputs.
        cache: Tuple of values (x, sample_mean, sample_var, x_hat, eps, gamma, beta) we stored.
    """

    out, cache = None, None

    M, channels, height, width = x.shape
    x_new = x.transpose(0, 2, 3, 1).reshape(M*height*width, channels)
    out, cache = batchnorm_forward(x_new, gamma, beta, bn_param)
    out = out.reshape(M, height, width, channels).transpose(0, 3, 1, 2)

    return out, cache

def spatial_batchnorm_backward(dout, cache):
    """Spatial backward propagation of batch normalization.

    Args:
        dout: The derivative with respect to last variable.
        cache: Tuple of values (x, sample_mean, sample_var, x_hat, eps, gamma, beta)
               we stored in forward propagation.

    Returns:
        dx: The derivative with respect to activation output.
        dgamma: The derivative with respect to gamma.
        dbeta: The derivative with respect to beta.
    """

    dx, dgamma, dbeta = None, None, None

    M, channels, height, width = dout.shape
    dout_new = dout.transpose(0, 2, 3, 1).reshape(M*height*width, channels)
    dx, dgamma, dbeta = batchnorm_backward(dout_new, cache)
    dx = dx.reshape(M, height, width, channels).transpose(0, 3, 1, 2)

    return dx, dgamma, dbeta





def relu(x):
    """Relu activation function.

    Args:
        x: Inputs.

    Returns:
        Activation outputs.
        activation_cache: The cache of activation input.
    """
    activation_cache = x
    return np.maximum(x, 0), activation_cache

def relu_backward(dA, activation_cache):
    """Relu backward propagation.

    Args:
        dA: The derivative of last variable.
        activation_cache: The cache of activation input we stored in relu function.

    Returns:
        The derivative of activation input.
    """
    return np.multiply((activation_cache >= 0), 1) * dA

def softmax(x):
    """Softmax activation function.

    Args:
        x: Inputs.

    Returns:
        Activation outputs.
        activation_cache: The cache of activation input.
    """

    assert x.shape == (10, 128)
    activation_cache = x
    return np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True), activation_cache

def softmax_backward(y, yhat):
    """Softmax backward propagation.

    Args:
        y: The real labels.
        yhat: The predictions we got.

    Returns:
        The derivative of activation input.
    """

    return yhat - y





def initialize_parameters():
    """Initialize all parameters.

    Returns:
        parameters: A python dictionary of all parameters.
    """

    parameters = {}

    parameters['w_conv1'] = np.random.randn(5, 3, 3, 3) * 0.01
    parameters['b_conv1'] = np.zeros((5, 1))
    parameters['w_conv2'] = np.random.randn(5, 5, 3, 3) * 0.01
    parameters['b_conv2'] = np.zeros((5, 1))
    parameters['w1'] = np.random.randn(384, 320) * 0.01
    parameters['b1'] = np.zeros((384, 1))
    parameters['w2'] = np.random.randn(192, 384) * 0.01
    parameters['b2'] = np.zeros((192, 1))
    parameters['w3'] = np.random.randn(10, 192) * 0.01
    parameters['b3'] = np.zeros((10, 1))
    parameters['gamma_conv1'] = 1
    parameters['gamma_conv2'] = 1
    parameters['gamma1'] = 1
    parameters['gamma2'] = 1
    parameters['gamma3'] = 1
    parameters['beta_conv1'] = 0
    parameters['beta_conv2'] = 0
    parameters['beta1'] = 0
    parameters['beta2'] = 0
    parameters['beta3'] = 0

    return parameters

def update_parameters(parameters, grads, learning_rate):
    """Update parameters using gradient descent.

    Args:
        parameters: python dictionary containing your parameters.
        grads: python dictionary containing your gradients, output of L_model_backward.

    Returns:
        parameters: python dictionary containing your updated parameters.
                    parameters["W" + str(l)] = ...
                    parameters["b" + str(l)] = ...
    """

    # Update rule for each parameter.
    parameters['w_conv2'] = parameters['w_conv2'] - learning_rate * grads['dw_conv2']
    parameters['b_conv2'] = parameters['b_conv2'] - learning_rate * grads['db_conv2']
    parameters['w_conv1'] = parameters['w_conv1'] - learning_rate * grads['dw_conv1']
    parameters['b_conv1'] = parameters['b_conv1'] - learning_rate * grads['db_conv1']
    parameters['w3'] = parameters['w3'] - learning_rate * grads['dw3']
    parameters['b3'] = parameters['b3'] - learning_rate * grads['db3']
    parameters['w2'] = parameters['w2'] - learning_rate * grads['dw2']
    parameters['b2'] = parameters['b2'] - learning_rate * grads['db2']
    parameters['w1'] = parameters['w1'] - learning_rate * grads['dw1']
    parameters['b1'] = parameters['b1'] - learning_rate * grads['db1']
    parameters['gamma_conv2'] = parameters['gamma_conv2'] - learning_rate * \
                                                            grads['dgamma_conv2']
    parameters['beta_conv2'] = parameters['beta_conv2'] - learning_rate * \
                                                            grads['dbeta_conv2']
    parameters['gamma_conv1'] = parameters['gamma_conv1'] - learning_rate * \
                                                            grads['dgamma_conv1']
    parameters['beta_conv1'] = parameters['beta_conv1'] - learning_rate * \
                                                            grads['dbeta_conv1']
    parameters['gamma3'] = parameters['gamma3'] - learning_rate * grads['dgamma3']
    parameters['beta3'] = parameters['beta3'] - learning_rate * grads['dbeta3']
    parameters['gamma2'] = parameters['gamma2'] - learning_rate * grads['dgamma2']
    parameters['beta2'] = parameters['beta2'] - learning_rate * grads['dbeta2']
    parameters['gamma1'] = parameters['gamma1'] - learning_rate * grads['dgamma1']
    parameters['beta1'] = parameters['beta1'] - learning_rate * grads['dbeta1']

    return parameters





def preprocessing_inputs():
    """Construct input for CIFAR training.

    Returns:
        mini_batches: The whole training data which is split up to n mini_batches.

    Raises:
        ValueError: If no data_dir
    """
    if not data_dir:
        raise ValueError('Please supply a data_dir')

    return cifar10_input.preprocessing_inputs(data_dir=data_dir, batch_size=batch_size)

def preprocessing_inputs_test():
    """Construct input for CIFAR testing.

    Returns:
        test_images: A (10000, 3, 32, 32) shape testing images.
        test_labels: A (1, 10000) shape testing labels.

    Raises:
        ValueError: If no data_dir
    """
    if not data_dir:
        raise ValueError('Please supply a data_dir')

    return cifar10_input.preprocessing_inputs_test(data_dir=data_dir, batch_size=batch_size)




def gradient_check(images, labels, epsilon):
    """Gradient checking for parameters.

    Args:
        images: Training images.
        labels: Training labels.
        epsilon: A little real number closed to 0.

    Returns:
        difference: The difference between two derivatives computing ways.
    """

    parameters = initialize_parameters()
    w_conv1 = parameters['w_conv1']
    b_conv1 = parameters['b_conv1']
    w_conv2 = parameters['w_conv2']
    b_conv2 = parameters['b_conv2']
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    w3 = parameters['w3']
    b3 = parameters['b3']
    gamma_conv1 = parameters['gamma_conv1']
    gamma_conv2 = parameters['gamma_conv2']
    gamma1 = parameters['gamma1']
    gamma2 = parameters['gamma2']
    gamma3 = parameters['gamma3']
    beta_conv1 = parameters['beta_conv1']
    beta_conv2 = parameters['beta_conv2']
    beta1 = parameters['beta1']
    beta2 = parameters['beta2']
    beta3 = parameters['beta3']



    # Compute backward gradients
    # conv_1
    conv_param = {}
    conv_param['stride'] = 1
    conv_param['pad'] = 1
    bn_param = {}
    bn_param['mode'] = 'train'
    bn_param['eps'] = 1e-5
    bn_param['momentum'] = 0.9
    conv_1, cache_conv_1 = conv_forward_naive(images, w_conv1,
                                              b_conv1, conv_param)
    conv_1, cache_batchnorm_conv_1 = spatial_batchnorm_forward(conv_1,
                                                               gamma_conv1,
                                                               beta_conv1,
                                                               bn_param)
    conv_1, cache_activation_conv_1 = relu(conv_1)

    # pool_1
    pool_param = {}
    pool_param['pool_height'] = 2
    pool_param['pool_width'] = 2
    pool_param['stride'] = 2
    pool_1, cache_pool_1 = max_pool_forward_naive(conv_1, pool_param)

    # conv_2
    conv_param = {}
    conv_param['stride'] = 1
    conv_param['pad'] = 1
    bn_param = {}
    bn_param['mode'] = 'train'
    bn_param['eps'] = 1e-5
    bn_param['momentum'] = 0.9
    conv_2, cache_conv_2 = conv_forward_naive(pool_1, w_conv2, b_conv2,
                                              conv_param)
    conv_2, cache_batchnorm_conv_2 = spatial_batchnorm_forward(conv_2,
                                                               gamma_conv2,
                                                               beta_conv2,
                                                               bn_param)
    conv_2, cache_activation_conv_2 = relu(conv_2)

    # pool_2
    pool_param = {}
    pool_param['pool_height'] = 2
    pool_param['pool_width'] = 2
    pool_param['stride'] = 2
    pool_2, cache_pool_2 = max_pool_forward_naive(conv_2, pool_param)

    # local_3_relu
    reshape = pool_2.reshape(128, -1).T
    bn_param = {}
    bn_param['mode'] = 'train'
    bn_param['eps'] = 1e-5
    bn_param['momentum'] = 0.9
    local_3, cache_local_3 = linear_activation_forward(reshape, w1, b1,
                                                       activation='relu',
                                                       gamma=gamma1,
                                                       beta=beta1,
                                                       bn_param=bn_param)

    # local_4_relu
    bn_param['mode'] = 'train'
    bn_param['eps'] = 1e-5
    bn_param['momentum'] = 0.9
    local_4, cache_local_4 = linear_activation_forward(local_3, w2, b2,
                                                       activation='relu',
                                                       gamma=gamma2,
                                                       beta=beta2,
                                                       bn_param=bn_param)

    # local_5_softmax
    bn_param['mode'] = 'train'
    bn_param['eps'] = 1e-5
    bn_param['momentum'] = 0.9
    logits, cache_local_5 = linear_activation_forward(local_4, w3, b3,
                                                      activation='softmax',
                                                      gamma=gamma3,
                                                      beta=beta3,
                                                      bn_param=bn_param)

    # Compute cost
    cost, new_labels = compute_cost(logits, labels)

    # Backward prop local
    dZ = softmax_backward(new_labels, logits)
    dZ, dgamma3, dbeta3 = batchnorm_backward(dZ, cache_local_5[2])
    dA2, dw3, db3 = linear_backward(dZ, cache_local_5[0])
    dA1, dw2, db2, dgamma2, dbeta2 = linear_activation_backward(dA2,
                                                                cache_local_4)
    dA0, dw1, db1, dgamma1, dbeta1 = linear_activation_backward(dA1,
                                                                cache_local_3)

    # Backward prop conv
    dout_pool_2 = dA0.T.reshape(128, 5, 8, 8)
    # pool_2 back prop
    d_pool_2 = max_pool_backward_naive(dout_pool_2, cache_pool_2)

    # conv_2 back prop
    dout_conv_2 = relu_backward(d_pool_2, cache_activation_conv_2)
    dout_conv_2, dgamma_conv2, dbeta_conv2 = spatial_batchnorm_backward(dout_conv_2,
                                                                        cache_batchnorm_conv_2)
    dout_pool_1, dw_conv2, db_conv2 = conv_backward_naive(dout_conv_2,
                                                          cache_conv_2)

    # pool_1 back prop
    d_pool_1 = max_pool_backward_naive(dout_pool_1, cache_pool_1)

    # conv_1 back prop
    dout_conv_1 = relu_backward(d_pool_1, cache_activation_conv_1)
    dout_conv_1, dgamma_conv1, dbeta_conv1 = spatial_batchnorm_backward(dout_conv_1,
                                                                        cache_batchnorm_conv_1)
    dAA, dw_conv1, db_conv1 = conv_backward_naive(dout_conv_1, cache_conv_1)

    # Reshape gradients
    dw_conv2 = dw_conv2.reshape(1, 5*5*3*3)
    db_conv2 = db_conv2.reshape(1, 5*1)
    dw_conv1 = dw_conv1.reshape(1, 5*3*3*3)
    db_conv1 = db_conv1.reshape(1, 5*1)
    dw1 = dw1.reshape(1, 384*320)
    db1 = db1.reshape(1, 384*1)
    dw2 = dw2.reshape(1, 192*384)
    db2 = db2.reshape(1, 192*1)
    dw3 = dw3.reshape(1, 10*192)
    db3 = db3.reshape(1, 10*1)
    dgamma_conv2 = dgamma_conv2.reshape(1, 5)
    dbeta_conv2 = dbeta_conv2.reshape(1, 5)
    dgamma_conv1 = dgamma_conv1.reshape(1, 5)
    dbeta_conv1 = dbeta_conv1.reshape(1, 5)
    dgamma1 = dgamma1.reshape(1, 128)
    dbeta1 = dbeta1.reshape(1, 128)
    dgamma2 = dgamma2.reshape(1, 128)
    dbeta2 = dbeta2.reshape(1, 128)
    dgamma3 = dgamma3.reshape(1, 128)
    dbeta3 = dbeta3.reshape(1, 128)

    grads = np.concatenate((dw_conv2,db_conv2,dw_conv1,db_conv1,dw1,db1,dw2,db2,
                            dw3,db3,dgamma_conv2,dbeta_conv2,dgamma_conv1,dbeta_conv1,
                            dgamma1,dbeta1,dgamma2,dbeta2,dgamma3,dbeta3),axis=1)




    # Compute approximate gradient
    gradapprox = np.zeros((1, grads.shape[1]))

    # Reshape parameters
    w_conv2 = w_conv2.reshape(1, 5 * 5 * 3 * 3)
    b_conv2 = b_conv2.reshape(1, 5 * 1)
    w_conv1 = w_conv1.reshape(1, 5 * 3 * 3 * 3)
    b_conv1 = b_conv1.reshape(1, 5 * 1)
    w1 = w1.reshape(1, 384 * 320)
    b1 = b1.reshape(1, 384 * 1)
    w2 = w2.reshape(1, 192 * 384)
    b2 = b2.reshape(1, 192 * 1)
    w3 = w3.reshape(1, 10 * 192)
    b3 = b3.reshape(1, 10 * 1)
    gamma_conv2 = np.tile(np.array(gamma_conv2), 5).reshape(1, 5)
    beta_conv2 = np.tile(np.array(beta_conv2), 5).reshape(1, 5)
    gamma_conv1 = np.tile(np.array(gamma_conv1), 5).reshape(1, 5)
    beta_conv1 = np.tile(np.array(beta_conv1), 5).reshape(1, 5)
    gamma1 = np.tile(np.array(gamma1), 128).reshape(1, 128)
    beta1 = np.tile(np.array(beta1), 128).reshape(1, 128)
    gamma2 = np.tile(np.array(gamma2), 128).reshape(1, 128)
    beta2 = np.tile(np.array(beta2), 128).reshape(1, 128)
    gamma3 = np.tile(np.array(gamma3), 128).reshape(1, 128)
    beta3 = np.tile(np.array(beta3), 128).reshape(1, 128)

    collection = np.concatenate((w_conv2, b_conv2, w_conv1, b_conv1, w1, b1, w2, b2,
                            w3, b3, gamma_conv2, beta_conv2, gamma_conv1, beta_conv1,
                            gamma1, beta1, gamma2, beta2, gamma3, beta3), axis=1)

    collection_plus = collection
    collection_minus = collection

    for i in range(collection.shape[1]):

        collection_plus[:, i] = collection_plus[:, i] + epsilon
        collection_minus[:, i] = collection_minus[:, i] - epsilon

        # Plus parameters forward prop
        w_conv2 = collection_plus[:, 0: 225].reshape(5, 5, 3, 3)
        b_conv2 = collection_plus[:, 225: 230].reshape(5, 1)
        w_conv1 = collection_plus[:, 230: 365].reshape(5, 3, 3, 3)
        b_conv1 = collection_plus[:, 365: 370].reshape(5, 1)
        w1 = collection_plus[:, 370: 123250].reshape(384, 320)
        b1 = collection_plus[:, 123250: 123634].reshape(384, 1)
        w2 = collection_plus[:, 123634: 197362].reshape(192, 384)
        b2 = collection_plus[:, 197362: 197554].reshape(192, 1)
        w3 = collection_plus[:, 197554: 199474].reshape(10, 192)
        b3 = collection_plus[:, 199474: 199484].reshape(10, 1)
        gamma_conv2 = collection_plus[:, 199484: 199489].reshape(5,)
        beta_conv2 = collection_plus[:, 199489: 199494].reshape(5,)
        gamma_conv1 = collection_plus[:, 199494: 199499].reshape(5,)
        beta_conv1 = collection_plus[:, 199499: 199504].reshape(5,)
        gamma1 = collection_plus[:, 199504: 199632].reshape(128,)
        beta1 = collection_plus[:, 199632: 199760].reshape(128,)
        gamma2 = collection_plus[:, 199760: 199888].reshape(128,)
        beta2 = collection_plus[:, 199888: 200016].reshape(128,)
        gamma3 = collection_plus[:, 200016: 200144].reshape(128,)
        beta3 = collection_plus[:, 200144: 200272].reshape(128,)

        # conv_1
        conv_param = {}
        conv_param['stride'] = 1
        conv_param['pad'] = 1
        bn_param = {}
        bn_param['mode'] = 'train'
        bn_param['eps'] = 1e-5
        bn_param['momentum'] = 0.9
        conv_1, cache_conv_1 = conv_forward_naive(images, w_conv1,
                                                  b_conv1, conv_param)
        conv_1, cache_batchnorm_conv_1 = spatial_batchnorm_forward(conv_1,
                                                                   gamma_conv1,
                                                                   beta_conv1,
                                                                   bn_param)
        conv_1, cache_activation_conv_1 = relu(conv_1)

        # pool_1
        pool_param = {}
        pool_param['pool_height'] = 2
        pool_param['pool_width'] = 2
        pool_param['stride'] = 2
        pool_1, cache_pool_1 = max_pool_forward_naive(conv_1, pool_param)

        # conv_2
        conv_param = {}
        conv_param['stride'] = 1
        conv_param['pad'] = 1
        bn_param = {}
        bn_param['mode'] = 'train'
        bn_param['eps'] = 1e-5
        bn_param['momentum'] = 0.9
        conv_2, cache_conv_2 = conv_forward_naive(pool_1, w_conv2, b_conv2,
                                                  conv_param)
        conv_2, cache_batchnorm_conv_2 = spatial_batchnorm_forward(conv_2,
                                                                   gamma_conv2,
                                                                   beta_conv2,
                                                                   bn_param)
        conv_2, cache_activation_conv_2 = relu(conv_2)

        # pool_2
        pool_param = {}
        pool_param['pool_height'] = 2
        pool_param['pool_width'] = 2
        pool_param['stride'] = 2
        pool_2, cache_pool_2 = max_pool_forward_naive(conv_2, pool_param)

        # local_3_relu
        reshape = pool_2.reshape(128, -1).T
        bn_param = {}
        bn_param['mode'] = 'train'
        bn_param['eps'] = 1e-5
        bn_param['momentum'] = 0.9
        local_3, cache_local_3 = linear_activation_forward(reshape, w1, b1,
                                                           activation='relu',
                                                           gamma=gamma1,
                                                           beta=beta1,
                                                           bn_param=bn_param)

        # local_4_relu
        bn_param['mode'] = 'train'
        bn_param['eps'] = 1e-5
        bn_param['momentum'] = 0.9
        local_4, cache_local_4 = linear_activation_forward(local_3, w2, b2,
                                                           activation='relu',
                                                           gamma=gamma2,
                                                           beta=beta2,
                                                           bn_param=bn_param)

        # local_5_softmax
        bn_param['mode'] = 'train'
        bn_param['eps'] = 1e-5
        bn_param['momentum'] = 0.9
        logits, cache_local_5 = linear_activation_forward(local_4, w3, b3,
                                                          activation='softmax',
                                                          gamma=gamma3,
                                                          beta=beta3,
                                                          bn_param=bn_param)

        # Compute cost
        cost_plus, new_labels = compute_cost(logits, labels)



        # Minus parameters forward prop
        w_conv2 = collection_minus[:, 0: 225].reshape(5, 5, 3, 3)
        b_conv2 = collection_minus[:, 225: 230].reshape(5, 1)
        w_conv1 = collection_minus[:, 230: 365].reshape(5, 3, 3, 3)
        b_conv1 = collection_minus[:, 365: 370].reshape(5, 1)
        w1 = collection_minus[:, 370: 123250].reshape(384, 320)
        b1 = collection_minus[:, 123250: 123634].reshape(384, 1)
        w2 = collection_minus[:, 123634: 197362].reshape(192, 384)
        b2 = collection_minus[:, 197362: 197554].reshape(192, 1)
        w3 = collection_minus[:, 197554: 199474].reshape(10, 192)
        b3 = collection_minus[:, 199474: 199484].reshape(10, 1)
        gamma_conv2 = collection_minus[:, 199484: 199489].reshape(5, )
        beta_conv2 = collection_minus[:, 199489: 199494].reshape(5, )
        gamma_conv1 = collection_minus[:, 199494: 199499].reshape(5, )
        beta_conv1 = collection_minus[:, 199499: 199504].reshape(5, )
        gamma1 = collection_minus[:, 199504: 199632].reshape(128, )
        beta1 = collection_minus[:, 199632: 199760].reshape(128, )
        gamma2 = collection_minus[:, 199760: 199888].reshape(128, )
        beta2 = collection_minus[:, 199888: 200016].reshape(128, )
        gamma3 = collection_minus[:, 200016: 200144].reshape(128, )
        beta3 = collection_minus[:, 200144: 200272].reshape(128, )

        # conv_1
        conv_param = {}
        conv_param['stride'] = 1
        conv_param['pad'] = 1
        bn_param = {}
        bn_param['mode'] = 'train'
        bn_param['eps'] = 1e-5
        bn_param['momentum'] = 0.9
        conv_1, cache_conv_1 = conv_forward_naive(images, w_conv1,
                                                  b_conv1, conv_param)
        conv_1, cache_batchnorm_conv_1 = spatial_batchnorm_forward(conv_1,
                                                                   gamma_conv1,
                                                                   beta_conv1,
                                                                   bn_param)
        conv_1, cache_activation_conv_1 = relu(conv_1)

        # pool_1
        pool_param = {}
        pool_param['pool_height'] = 2
        pool_param['pool_width'] = 2
        pool_param['stride'] = 2
        pool_1, cache_pool_1 = max_pool_forward_naive(conv_1, pool_param)

        # conv_2
        conv_param = {}
        conv_param['stride'] = 1
        conv_param['pad'] = 1
        bn_param = {}
        bn_param['mode'] = 'train'
        bn_param['eps'] = 1e-5
        bn_param['momentum'] = 0.9
        conv_2, cache_conv_2 = conv_forward_naive(pool_1, w_conv2, b_conv2,
                                                  conv_param)
        conv_2, cache_batchnorm_conv_2 = spatial_batchnorm_forward(conv_2,
                                                                   gamma_conv2,
                                                                   beta_conv2,
                                                                   bn_param)
        conv_2, cache_activation_conv_2 = relu(conv_2)

        # pool_2
        pool_param = {}
        pool_param['pool_height'] = 2
        pool_param['pool_width'] = 2
        pool_param['stride'] = 2
        pool_2, cache_pool_2 = max_pool_forward_naive(conv_2, pool_param)

        # local_3_relu
        reshape = pool_2.reshape(128, -1).T
        bn_param = {}
        bn_param['mode'] = 'train'
        bn_param['eps'] = 1e-5
        bn_param['momentum'] = 0.9
        local_3, cache_local_3 = linear_activation_forward(reshape, w1, b1,
                                                           activation='relu',
                                                           gamma=gamma1,
                                                           beta=beta1,
                                                           bn_param=bn_param)

        # local_4_relu
        bn_param['mode'] = 'train'
        bn_param['eps'] = 1e-5
        bn_param['momentum'] = 0.9
        local_4, cache_local_4 = linear_activation_forward(local_3, w2, b2,
                                                           activation='relu',
                                                           gamma=gamma2,
                                                           beta=beta2,
                                                           bn_param=bn_param)

        # local_5_softmax
        bn_param['mode'] = 'train'
        bn_param['eps'] = 1e-5
        bn_param['momentum'] = 0.9
        logits, cache_local_5 = linear_activation_forward(local_4, w3, b3,
                                                          activation='softmax',
                                                          gamma=gamma3,
                                                          beta=beta3,
                                                          bn_param=bn_param)

        # Compute cost
        cost_minus, new_labels = compute_cost(logits, labels)


        collection_plus = collection
        collection_minus = collection

        gradapprox[:, i] = (cost_plus - cost_minus) / (2 * epsilon)
        print cost_plus, cost_minus
        print grads[:, i], gradapprox[:, i]

        if i % 1 == 0:
            print ("step %i finished" % i)


    # Compute difference
    grads = grads.reshape(grads.shape[1], )
    gradapprox = gradapprox.reshape(gradapprox.shape[1], )
    numerator = np.linalg.norm(grads-gradapprox, ord=2)
    denominator = np.linalg.norm(grads, ord=2) + np.linalg.norm(gradapprox, ord=2)  # Step 2'
    difference = numerator / denominator

    print difference

    if difference < 1e-7:
        print "perfectly gradient computing!"
    else:
        print "some gradients wrong!"




def inference(mini_batches, learning_rate, num_iterations):
    """Build the CIFAR-10 model.

    Args:
        mini_batches: Input data which is split up to several mini_batches.
        learning_rate: Step length of gradient descent.
        num_iterations: Number of iterations.

    Returns:
        parameters: Updated parameters.
        costs: A python list of cost of every iteration.
        bn_param: A python dictionary of all the bn_params.
    """

    grads = {}
    costs = []
    parameters = initialize_parameters()
    w_conv1 = parameters['w_conv1']
    b_conv1 = parameters['b_conv1']
    w_conv2 = parameters['w_conv2']
    b_conv2 = parameters['b_conv2']
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    w3 = parameters['w3']
    b3 = parameters['b3']
    gamma_conv1 = parameters['gamma_conv1']
    gamma_conv2 = parameters['gamma_conv2']
    gamma1 = parameters['gamma1']
    gamma2 = parameters['gamma2']
    gamma3 = parameters['gamma3']
    beta_conv1 = parameters['beta_conv1']
    beta_conv2 = parameters['beta_conv2']
    beta1 = parameters['beta1']
    beta2 = parameters['beta2']
    beta3 = parameters['beta3']

    # Python dictionaries of {mode, epsilon, momentum, running_mean, running_var}.
    bn_param_conv_1 = {}
    bn_param_conv_1['mode'] = 'train'
    bn_param_conv_1['eps'] = 1e-5
    bn_param_conv_1['momentum'] = 0.9

    bn_param_conv_2 = bn_param_conv_1

    bn_param_local_3 = {}
    bn_param_local_3['mode'] = 'train'
    bn_param_local_3['eps'] = 1e-5
    bn_param_local_3['momentum'] = 0.9

    bn_param_local_4 = bn_param_local_3
    bn_param_local_5 = bn_param_local_3

    # Number of mini_batches
    num_complete_minibatches = len(mini_batches)

    # Start training
    for i in range(num_iterations):

        for j in range(1):

            # Calculate computing time during 1 mini_batch
            start_time = time.time()

            # conv_1
            conv_param = {}
            conv_param['stride'] = 1
            conv_param['pad'] = 1
            conv_1, cache_conv_1 = conv_forward_naive(mini_batches[j][0], w_conv1,
                                                      b_conv1, conv_param)
            conv_1, cache_batchnorm_conv_1 = spatial_batchnorm_forward(conv_1,
                                                                       gamma_conv1,
                                                                       beta_conv1,
                                                                       bn_param_conv_1)
            conv_1, cache_activation_conv_1 = relu(conv_1)


            # pool_1
            pool_param = {}
            pool_param['pool_height'] = 2
            pool_param['pool_width'] = 2
            pool_param['stride'] = 2
            pool_1, cache_pool_1 = max_pool_forward_naive(conv_1, pool_param)

            # conv_2
            conv_param = {}
            conv_param['stride'] = 1
            conv_param['pad'] = 1
            conv_2, cache_conv_2 = conv_forward_naive(pool_1, w_conv2, b_conv2,
                                                      conv_param)
            conv_2, cache_batchnorm_conv_2 = spatial_batchnorm_forward(conv_2,
                                                                       gamma_conv2,
                                                                       beta_conv2,
                                                                       bn_param_conv_2)
            conv_2, cache_activation_conv_2 = relu(conv_2)

            # pool_2
            pool_param = {}
            pool_param['pool_height'] = 2
            pool_param['pool_width'] = 2
            pool_param['stride'] = 2
            pool_2, cache_pool_2 = max_pool_forward_naive(conv_2, pool_param)

            # local_3_relu
            reshape = pool_2.reshape(128, -1).T
            local_3, cache_local_3 = linear_activation_forward(reshape, w1, b1,
                                                               activation='relu',
                                                               gamma=gamma1,
                                                               beta=beta1,
                                                               bn_param=bn_param_local_3)

            # local_4_relu
            local_4, cache_local_4 = linear_activation_forward(local_3, w2, b2,
                                                               activation='relu',
                                                               gamma=gamma2,
                                                               beta=beta2,
                                                               bn_param=bn_param_local_4)

            # local_5_softmax
            logits, cache_local_5 = linear_activation_forward(local_4, w3, b3,
                                                              activation='softmax',
                                                              gamma=gamma3,
                                                              beta=beta3,
                                                              bn_param=bn_param_local_5)

            # Compute cost
            cost, new_labels = compute_cost(logits, mini_batches[j][1])


            # Softmax/local5 backward propagation
            dZ = softmax_backward(new_labels, logits)
            dZ, dgamma3, dbeta3 = batchnorm_backward(dZ, cache_local_5[2])
            dA2, dw3, db3 = linear_backward(dZ, cache_local_5[0])

            # local4 backward propagation
            dA1, dw2, db2, dgamma2, dbeta2 = linear_activation_backward(dA2,
                                                                        cache_local_4)

            # local5 backward propagation
            dA0, dw1, db1, dgamma1, dbeta1 = linear_activation_backward(dA1,
                                                                        cache_local_3)

            # Reshape input of convolutional backward propagation
            dout_pool_2 = dA0.T.reshape(128, 5, 8, 8)


            # pool_2 backward propagation
            d_pool_2 = max_pool_backward_naive(dout_pool_2, cache_pool_2)


            # conv_2 backward propagation
            dout_conv_2 = relu_backward(d_pool_2, cache_activation_conv_2)
            dout_conv_2, dgamma_conv2, dbeta_conv2 = spatial_batchnorm_backward(dout_conv_2,
                                                                                cache_batchnorm_conv_2)
            dout_pool_1, dw_conv2, db_conv2 = conv_backward_naive(dout_conv_2,
                                                                  cache_conv_2)

            # pool_1 backward propagation
            d_pool_1 = max_pool_backward_naive(dout_pool_1, cache_pool_1)


            # conv_1 backward propagation
            dout_conv_1 = relu_backward(d_pool_1, cache_activation_conv_1)
            dout_conv_1, dgamma_conv1, dbeta_conv1 = spatial_batchnorm_backward(dout_conv_1,
                                                                                cache_batchnorm_conv_1)
            dAA, dw_conv1, db_conv1 = conv_backward_naive(dout_conv_1, cache_conv_1)


            # Store gradients
            grads['dw_conv2'] = dw_conv2
            grads['db_conv2'] = db_conv2
            grads['dw_conv1'] = dw_conv1
            grads['db_conv1'] = db_conv1
            grads['dw1'] = dw1
            grads['db1'] = db1
            grads['dw2'] = dw2
            grads['db2'] = db2
            grads['dw3'] = dw3
            grads['db3'] = db3
            grads['dgamma_conv2'] = dgamma_conv2
            grads['dbeta_conv2'] = dbeta_conv2
            grads['dgamma_conv1'] = dgamma_conv1
            grads['dbeta_conv1'] = dbeta_conv1
            grads['dgamma1'] = dgamma1
            grads['dbeta1'] = dbeta1
            grads['dgamma2'] = dgamma2
            grads['dbeta2'] = dbeta2
            grads['dgamma3'] = dgamma3
            grads['dbeta3'] = dbeta3


            # Update parameters in each layer
            parameters = update_parameters(parameters, grads, learning_rate)
            w_conv1 = parameters['w_conv1']
            b_conv1 = parameters['b_conv1']
            w_conv2 = parameters['w_conv2']
            b_conv2 = parameters['b_conv2']
            w1 = parameters['w1']
            b1 = parameters['b1']
            w2 = parameters['w2']
            b2 = parameters['b2']
            w3 = parameters['w3']
            b3 = parameters['b3']
            gamma_conv1 = parameters['gamma_conv1']
            gamma_conv2 = parameters['gamma_conv2']
            gamma1 = parameters['gamma1']
            gamma2 = parameters['gamma2']
            gamma3 = parameters['gamma3']
            beta_conv1 = parameters['beta_conv1']
            beta_conv2 = parameters['beta_conv2']
            beta1 = parameters['beta1']
            beta2 = parameters['beta2']
            beta3 = parameters['beta3']

            # Print costs
            duration = time.time() - start_time
            examples_per_sec = 128 / duration
            sec_per_batch = float(duration)
            step = i * num_complete_minibatches + j

            if (i * num_complete_minibatches + j) % 1 == 0:
                format_str = ('%s: step %d, cost = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step, cost,
                                     examples_per_sec, sec_per_batch))
            if step % 1 == 0:
                costs.append(cost)

    # A python dictionary of all the bn_params
    bn_param = {'bn_param_conv_1': bn_param_conv_1,
                'bn_param_conv_2': bn_param_conv_2,
                'bn_param_local_3': bn_param_local_3,
                'bn_param_local_4': bn_param_local_4,
                'bn_param_local_5': bn_param_local_5}

    return parameters, costs, bn_param




def compute_cost(logits, labels):
    """Compute cost function.

    Args:
        logits: Predictions.
        labels: True labels.

    Returns:
        cost: Cost function values.
        new_labels: Reshaped predictions.
    """

    assert logits.shape == (10, 128)
    assert labels.shape == (1, 128)

    new_labels = np.zeros((10, 128))
    for i in range(128):
        new_labels[labels[0][i]][i] = 1

    loss = np.sum(np.multiply(new_labels, np.log(logits)), axis=0)
    cost = -np.mean(loss)

    # return total_loss
    return cost, new_labels




def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    dest_directory = data_dir

    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)

    if not os.path.exists(filepath):

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
