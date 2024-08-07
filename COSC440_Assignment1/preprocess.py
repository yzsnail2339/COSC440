import gzip, os
import numpy as np
def get_data(inputs_file_path, labels_file_path, num_examples):
    """
    Takes in an inputs file path and labels file path, unzips both files,
    normalizes the inputs, and returns (NumPy array of inputs, NumPy
    array of labels). Read the data of the file into a buffer and use
    np.frombuffer to turn the data into a NumPy array. Keep in mind that
    each file has a header of a certain size. This method should be called
    within the main function of the assignment.py file to get BOTH the train and
    test data. If you change this method and/or write up separate methods for
    both train and test data, we will deduct points.

    Hint: look at the writeup for sample code on using the gzip library

    :param inputs_file_path: file path for inputs, something like
    'MNIST_data/t10k-images-idx3-ubyte.gz'
    :param labels_file_path: file path for labels, something like
    'MNIST_data/t10k-labels-idx1-ubyte.gz'
    :param num_examples: used to read from the bytestream into a buffer. Rather
    than hardcoding a number to read from the bytestream, keep in mind that each image
    (example) is 28 * 28, with a header of a certain number.
    :return: NumPy array of inputs as float32 and labels as int8
    """
    # TODO: Load inputs and labels
    with gzip.open(inputs_file_path, 'rb') as f:
        buffer = f.read(16 + num_examples * 784)
        images = np.frombuffer(buffer, dtype = np.uint8, count = -1, offset = 16).reshape(num_examples, 784)
        
    with gzip.open(labels_file_path, 'rb') as l:
        buffer = l.read(8 + num_examples)
        labels = np.frombuffer(buffer, dtype = np.uint8, count = -1, offset = 8)
    # TODO: Normalize inputs
    images = images.astype(np.float32) / 255
    labels = labels.astype(np.int8)
    return images, labels