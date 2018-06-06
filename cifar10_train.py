"""Train CIFAR-10 using a single CPU."""
import numpy as np
import matplotlib.pyplot as plt

import cifar10
import cifar10_eval


learning_rate = 0.01 # Step length of gradient descent
num_iterations = 2 # Number of iterations


def train():
    """Train CIFAR-10 for a number of steps."""

    # Get training set
    mini_batches = cifar10.preprocessing_inputs()


    # Training
    parameters, costs, bn_param = cifar10.inference(mini_batches, learning_rate=learning_rate,
                                                    num_iterations=num_iterations)


    # Get training and testing images and labels
    minibatch_test = cifar10.preprocessing_inputs_test()
    num_minibatch_test = len(minibatch_test)
    minibatch_train = cifar10.preprocessing_inputs()
    num_minibatch_train = len(minibatch_train)


    # Predict test/train set examples
    Y_prediction_test = cifar10_eval.predict(parameters, minibatch_test, bn_param)
    Y_prediction_train = cifar10_eval.predict(parameters, minibatch_train, bn_param)


    # Print train/test Errors
    train_right = 0.; test_right = 0.
    for i in range(num_minibatch_train):
        for j in range(128):
            if Y_prediction_train[i][j] == minibatch_train[i][1][0][j]: train_right += 1
    for i in range(num_minibatch_test):
        for j in range(128):
            if Y_prediction_test[i][j] == minibatch_test[i][1][0][j]: test_right += 1
    print("train accuracy: {} %".format(train_right/100))
    print("test accuracy: {} %".format(test_right/100))


    # Plot cost function iteration image
    costs = np.squeeze(costs)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


if __name__ == '__main__':
    train()
