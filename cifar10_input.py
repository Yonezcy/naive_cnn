import numpy as np
import os


# Number of data_batch files
num_files = 5

# Number of training samples
num_samples = 50000

# Number of testing samples
num_test_samples = 10000


def unpickle(file):
    """Load dataSet from target directory.

    Args:
        file: target "data_batch" file.

    Returns:
        A dict contains data, labels and other attributes.
    """
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


def read_cifar10(filenames):
    """Reads and parses examples from CIFAR10 data files.

     Args:
        filenames: A String array of target filenames.

     Returns:
        train_images: A (50000, 3072) shape training images.
        train_labels: A (50000,) shape training labels.
     """

    # Read the dataSet and split it up to images and labels.
    dict_train_batch = []

    # Read files
    for i in range(num_files):
        dict_train_batch.append(unpickle(filenames[i]))

    # Get training data
    train_images = np.concatenate((dict_train_batch[0]['data'],
                                   dict_train_batch[1]['data'],
                                   dict_train_batch[2]['data'],
                                   dict_train_batch[3]['data'],
                                   dict_train_batch[4]['data']))

    # Get training label
    train_labels = np.concatenate((dict_train_batch[0]['labels'],
                                   dict_train_batch[1]['labels'],
                                   dict_train_batch[2]['labels'],
                                   dict_train_batch[3]['labels'],
                                   dict_train_batch[4]['labels']))

    return train_images, train_labels


def read_cifar10_test(filename):
    """Reads and parses examples from CIFAR10 data files.

     Args:
        filename: Target filename.

     Returns:
        test_images: A (10000, 3072) shape testing images.
        test_labels: A (10000,) shape testing labels.
     """

    # Read the dataSet and split it up to images and labels.
    dict_test_batch = unpickle(filename)

    # Get training data
    test_images = dict_test_batch['data']

    # Get training label
    test_labels = dict_test_batch['labels']

    return test_images, test_labels


def _generate_image_and_label_batch(image, label, batch_size):
    """Construct mini_batches of images and labels.

    Args:
        image: A (50000, 3, 32, 32) shape training images.
        label: A (1, 50000) shape training labels.
        batch_size: Number of images per batch.

    Returns:
        mini_batches: The whole training data which is split up to n mini_batches.
    """

    # Generate the whole batches of one epoch
    m = image.shape[0]
    mini_batches = []

    # Random permutation for samples
    permutation = list(np.random.permutation(m))
    shuffled_x = image[permutation, :, :, :]
    shuffled_y = label[:, permutation]

    # Split it up to mini_batches
    num_complete_minibatches = m / batch_size
    for k in range(0, num_complete_minibatches):
        mini_batch_x = shuffled_x[(k*batch_size):((k+1)*batch_size), :, :, :]
        mini_batch_y = shuffled_y[:, (k*batch_size):((k+1)*batch_size)]
        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)

    return mini_batches


def preprocessing_inputs(data_dir, batch_size, mode):
    """Construct input for CIFAR training.

    Args:
        data_dir: Path to the CIFAR-10 data directory.
        batch_size: Number of images per batch.

    Returns:
        mini_batches: The whole training data which is split up to n mini_batches.
    """

    # Three dimensions of image
    height = 32
    width = 32
    depth = 3

    # Check whether the files exist
    if not data_dir:
        raise ValueError('Please supply a data_dir')
    filenames = [os.path.join(data_dir, 'data_batch_%d' % i)
                 for i in range(1, 6)]

    # Get training data
    train_images, train_labels = read_cifar10(filenames)

    # Subtract off the mean and divide by the variance of the pixels
    train_images = (train_images - np.mean(train_images, axis=1, keepdims=True)) / \
                    np.std(train_images, axis=1, keepdims=True)

    # Reshape images shape (50000, 3072) to (50000, 3, 32, 32)
    train_images = train_images.reshape(num_samples, depth, height, width)

    # Reshape labels shape (50000,) to (1, 50000)
    train_labels = train_labels.reshape(train_labels.shape[0], 1).T

    if mode == 'train':
        return _generate_image_and_label_batch(train_images, train_labels, batch_size)

    elif mode == 'evaluate':
        return train_images, train_labels

def preprocessing_inputs_test(data_dir):
    """Construct input for CIFAR training.

    Args:
        data_dir: Path to the CIFAR-10 data directory.
        batch_size: Number of images per batch.

    Returns:
        test_images: A (10000, 3, 32, 32) shape testing images.
        test_labels: A (1, 10000) shape testing labels.
    """

    # Three dimensions of image
    height = 32
    width = 32
    depth = 3

    # Check whether the files exist
    if not data_dir:
        raise ValueError('Please supply a data_dir')
    filename = os.path.join(data_dir, 'test_batch')

    # Get training data
    test_images, test_labels = read_cifar10_test(filename)

    # Subtract off the mean and divide by the variance of the pixels
    test_images = (test_images - np.mean(test_images, axis=1, keepdims=True)) / \
                    np.std(test_images, axis=1, keepdims=True)

    # Reshape images shape (10000, 3072) to (10000, 3, 32, 32)
    test_images = test_images.reshape(num_test_samples, depth, height, width)

    # Reshape labels shape (10000,) to (1, 10000)
    test_labels = test_labels.reshape(test_labels.shape[0], 1).T

    return test_images, test_labels


# Just for test
'''
if __name__ == '__main__':
    result = preprocessing_inputs('/Users/apple/desktop/cifar-10-batches-py', 128)
    print result
'''
