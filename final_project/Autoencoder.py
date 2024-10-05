import tensorflow as tf
import numpy as np

class Encoder(tf.keras.layers.Layer):
    def __init__(self):
       super(Encoder, self).__init__()
       self.encoder_conv_1 = tf.keras.layers.Conv2D(
           filters=8,
           kernel_size=3,
           strides=2,
           padding='same',
           kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
           activation=tf.keras.layers.LeakyReLU(alpha=0.2)
       )
       self.encoder_conv_2 = tf.keras.layers.Conv2D(
           filters=16,
           kernel_size=3,
           strides=2,
           padding='same',
           kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
           activation=tf.keras.layers.LeakyReLU(alpha=0.2)
       )
       self.encoder_conv_3 = tf.keras.layers.Conv2D(
           filters=32,
           kernel_size=3,
           strides=2,
           padding='same',
           kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
           activation=tf.keras.layers.LeakyReLU(alpha=0.2)
       )

    @tf.function
    def call(self, images):
      x = self.encoder_conv_1(images)
      x = self.encoder_conv_2(x)
      x = self.encoder_conv_3(x)
      return x
    
class Decoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder_deconv_1 = tf.keras.layers.Conv2DTranspose(
            filters=16,
            kernel_size=3,
            strides=2,
            padding='same',
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            activation=tf.keras.layers.LeakyReLU(alpha=0.2)
        )
        self.decoder_deconv_2 = tf.keras.layers.Conv2DTranspose(
            filters=8,
            kernel_size=3,
            strides=2,
            padding='same',
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            activation=tf.keras.layers.LeakyReLU(alpha=0.2)
        )
        self.decoder_deconv_3 = tf.keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=3,
            strides=2,
            padding='same',
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            activation=tf.keras.layers.LeakyReLU(alpha=0.2)
        )
    @tf.function
    def call(self, encoder_output):
      x = self.decoder_deconv_1(encoder_output)
      x = self.decoder_deconv_2(x)
      x = self.decoder_deconv_3(x)
      return x
    
class AutoEncoder(tf.keras.Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    @tf.function
    def call(self, inputs):
      encoded = self.encoder(inputs)
      decoded = self.decoder(encoded)
      return decoded