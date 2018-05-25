import numpy as np
import cifar10


def predict(parameters, image_test, bn_param):
    """Predict whether the label is 0 or 1 using parameters.

    Args:
        parameters: All the parameters through full training.
        image_test: Test images.
        bn_param: A python dictionary of all the bn_params.

    Returns:
        Y_prediction: All predictions (0/1) for the examples in image_test.
    """

    # Get parameters
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

    # Get bn_params
    bn_param_conv_1 = bn_param['bn_param_conv_1']
    bn_param_conv_1['mode'] = 'test'
    bn_param_conv_2 = bn_param['bn_param_conv_2']
    bn_param_conv_2['mode'] = 'test'
    bn_param_local_3 = bn_param['bn_param_local_3']
    bn_param_local_3['mode'] = 'test'
    bn_param_local_4 = bn_param['bn_param_local_4']
    bn_param_local_4['mode'] = 'test'
    bn_param_local_5 = bn_param['bn_param_local_5']
    bn_param_local_5['mode'] = 'test'


    # conv_1
    conv_param = {}
    conv_param['stride'] = 1
    conv_param['pad'] = 1
    conv_1, cache_conv_1 = cifar10.conv_forward_naive(image_test, w_conv1,
                                              b_conv1, conv_param)
    conv_1, cache_batchnorm_conv_1 = cifar10.spatial_batchnorm_forward(conv_1,
                                                               gamma_conv1,
                                                               beta_conv1,
                                                               bn_param_conv_1)
    conv_1, cache_activation_conv_1 = cifar10.relu(conv_1)

    # pool_1
    pool_param = {}
    pool_param['pool_height'] = 2
    pool_param['pool_width'] = 2
    pool_param['stride'] = 2
    pool_1, cache_pool_1 = cifar10.max_pool_forward_naive(conv_1, pool_param)

    # conv_2
    conv_param = {}
    conv_param['stride'] = 1
    conv_param['pad'] = 1
    conv_2, cache_conv_2 = cifar10.conv_forward_naive(pool_1, w_conv2, b_conv2,
                                              conv_param)
    conv_2, cache_batchnorm_conv_2 = cifar10.spatial_batchnorm_forward(conv_2,
                                                               gamma_conv2,
                                                               beta_conv2,
                                                               bn_param_conv_2)
    conv_2, cache_activation_conv_2 = cifar10.relu(conv_2)

    # pool_2
    pool_param = {}
    pool_param['pool_height'] = 2
    pool_param['pool_width'] = 2
    pool_param['stride'] = 2
    pool_2, cache_pool_2 = cifar10.max_pool_forward_naive(conv_2, pool_param)

    # local_3_relu
    reshape = pool_2.reshape(128, -1).T
    local_3, cache_local_3 = cifar10.linear_activation_forward(reshape, w1, b1,
                                                       activation='relu',
                                                       gamma=gamma1,
                                                       beta=beta1,
                                                       bn_param=bn_param_local_3)

    # local_4_relu
    local_4, cache_local_4 = cifar10.linear_activation_forward(local_3, w2, b2,
                                                       activation='relu',
                                                       gamma=gamma2,
                                                       beta=beta2,
                                                       bn_param=bn_param_local_4)

    # local_5_softmax
    logits, cache_local_5 = cifar10.linear_activation_forward(local_4, w3, b3,
                                                      activation='softmax',
                                                      gamma=gamma3,
                                                      beta=beta3,
                                                      bn_param=bn_param_local_5)
    Y_prediction = np.zeros((logits.shape[0], 1))

    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            if logits[i][j] == np.max(logits[i]):
                Y_prediction[i][0] = i



    assert (Y_prediction.shape == (logits.shape[0], 1))

    return Y_prediction
