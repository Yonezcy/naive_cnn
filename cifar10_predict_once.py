"""Prediction for CIFAR-10."""
import numpy as np
from PIL import Image
import matplotlib.image as mpimg

import cifar10


learning_rate = 0.01 # Step length of gradient descent
num_iterations = 1 # Number of iterations

height = 32; width = 32; channels = 3 # Size of CIFAR-10 image

location = '/Users/apple/desktop/car/image_0001.jpg'
saved_location = '/Users/apple/desktop/1.jpg'


def resize_image(location):
    """Resize image to 32*32.

    Args:
        location: Saved location of image.
    """
    im = Image.open(location)
    im_resize = im.resize((height, width))
    im_resize.save(saved_location)


def preprocessing(location, saved_location):
    """Resize image and load image.

    Args:
        location: Saved location of image.
        saved_location: New location which new image want to save in.

    Returns:
        image: image of shape[height, width, channels].
    """
    resize_image(location)
    image = mpimg.imread(saved_location)
    image = image.reshape(height*width, channels).T
    image = image.reshape(1, channels, height, width)
    return image


def predict_once():
    """Predict class of new image."""

    # Preprocessing image
    image = preprocessing(location, saved_location)

    # Get training set
    mini_batches = cifar10.preprocessing_inputs()

    # Training
    parameters, costs, bn_param = cifar10.inference(mini_batches, learning_rate=learning_rate,
                                                    num_iterations=num_iterations)

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
    bn_param_conv_2 = bn_param['bn_param_conv_2']
    bn_param_local_3 = bn_param['bn_param_local_3']
    bn_param_local_4 = bn_param['bn_param_local_4']
    bn_param_local_5 = bn_param['bn_param_local_5']

    # conv_1
    conv_param = {}
    conv_param['stride'] = 1
    conv_param['pad'] = 1
    conv_1, cache_conv_1 = cifar10.conv_forward_naive(image, w_conv1,
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
    reshape = pool_2.reshape(1, -1).T
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

    dict = {0: "airplane",
            1: "automobile",
            2: "bird",
            3: "cat",
            4: 'deer',
            5: 'dog',
            6: 'frog',
            7: 'horse',
            8: 'sheep',
            9: 'truck'}

    for i in range(10):
        if logits[:, 0][i] == np.max(logits[:, 0]):
            print "class of image is " + dict[i]


if __name__ == '__main__':
    predict_once()

