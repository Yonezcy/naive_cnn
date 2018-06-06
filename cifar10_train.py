"""Train CIFAR-10 using a single CPU."""
import numpy as np
import matplotlib.pyplot as plt

import cifar10
import cifar10_eval


learning_rate = 0.01 # Step length of gradient descent
num_iterations = 1 # Number of iterations


def train():
    """Train CIFAR-10 for a number of steps."""

    # Get training set
    mini_batches = cifar10.preprocessing_inputs('train')


    # Training
    parameters, costs, bn_param = cifar10.inference(mini_batches, learning_rate=learning_rate,
                                                    num_iterations=num_iterations)


    # Get training and testing images and labels
    test_images, test_labels = cifar10.preprocessing_inputs_test()
    train_images, train_labels = cifar10.preprocessing_inputs('evaluate')


    # Predict test/train set examples
    Y_prediction_test = cifar10_eval.predict(parameters, test_images, bn_param, batch_size=10000)
    Y_prediction_train = cifar10_eval.predict(parameters, train_images, bn_param, batch_size=128)


    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train[0] - train_labels[0])) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test[0] - test_labels[0])) * 100))


    # Plot cost function iteration image
    costs = np.squeeze(costs)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


if __name__ == '__main__':
    train()
