import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

class SuperSampler(tf.keras.layers.Layer):
    def __init__(self):
        # TODO Create the upsampling convolution layers
        super(SuperSampler, self).__init__()
        self.upsampling_convolution_1 = tf.keras.layers.Conv2D(
                filters=8,
                kernel_size=3,
                strides=1,
                padding='same',
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                activation=tf.keras.layers.LeakyReLU(alpha=0.2)
                )
        # self.upsampling_convolution_2 = tf.keras.Sequential([
        #     tf.keras.layers.Lambda(lambda x: tf.image.resize(x, size=(128,128), method='nearest')),
        #     tf.keras.layers.Conv2D(
        #         filters = 16,
        #         kernel_size = 3,
        #         strides = 1,
        #         padding = 'same',
        #         kernel_initializer = tf.keras.initializers.RandomNormal(stddev=0.1),
        #         activation = tf.keras.layers.LeakyReLU(alpha=0.2)
        #     )
        # ])
        # self.upsampling_convolution_3 = tf.keras.Sequential([
        #     tf.keras.layers.Lambda(lambda x: tf.image.resize(x, size=(256,256), method='nearest')),
        #     tf.keras.layers.Conv2D(
        #         filters = 3,
        #         kernel_size = 3,
        #         strides = 1,
        #         padding = 'same',
        #         kernel_initializer = tf.keras.initializers.RandomNormal(stddev=0.1),
        #         activation = tf.keras.layers.LeakyReLU(alpha=0.2)
        #     )
        # ])

    @tf.function
    def call(self, input):
        # TODO Implement the call function
        input = tf.image.resize(input, size=(64,64),method='nearest')
        x =  self.upsampling_convolution_1(input)
        # x = self.upsampling_convolution_2(x)
        # x = self.upsampling_convolution_3(x)
        return x

# Below are some basic test assertions. Please note that your code will be
# evaluated using external tests for marking.

example_batch_size = 100
example_images = np.ones((example_batch_size, 32, 32, 3))

network = SuperSampler()
outputs = network.call(example_images)
# assert(outputs.shape[0] == 100)
# assert(outputs.shape[1] == 256)
# assert(outputs.shape[2] == 256)
# assert(outputs.shape[3] == 3)
print(outputs.shape)