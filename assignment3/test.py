import tensorflow as tf
import numpy as np

tf.random.set_seed(1337)  # setting seed for later part of assignment. DO NOT CHANGE

import tensorflow as tf
import numpy as np

tf.random.set_seed(1337)  # setting seed for later part of assignment. DO NOT CHANGE

print(tf.__version__)
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

"""
This function adds no noise, we use this as a way of comparing the
autoencoder when there's no input noise
"""
def no_noise(x):
  # TODO
  return tf.cast(x, tf.float32)

"""
This function should add a random uniform tensor of the shape of x between
-0.3 to 0.3 to x.

It should then "clip" x to between 0 and 1 (hint: check out tf.clip_by_value)
"""
def random_noise(x):
  # TODO
  x = tf.cast(x, tf.float32)
  noise = tf.Variable(tf.random.uniform(x.shape,
                minval=-0.3,
                maxval=0.3,
                dtype=tf.float32),
                name = "noise")
  return tf.clip_by_value(x + noise, 0, 1)

"""
This function should multiply a random uniform tensor of the shape of x between
0 to 2.0 to x.

It should then "clip" x to between 0 and 1 (hint: check out tf.clip_by_value)
"""
def random_scale(x):
  # TODO
  x = tf.cast(x, tf.float32)
  noise = tf.Variable(tf.random.uniform(x.shape,
                minval=0,
                maxval=2.0,
                dtype=tf.float32),
                name = "noise")
  return tf.clip_by_value(x + noise, 0, 1)

# some "unit tests"
x = [[0.3,0.1],[0.2,0]]
y = [[0.1,1],[0,0.33]]
result_1 = random_noise(x)
result_2 = random_scale(x)
result_3 = random_noise(y)
result_4 = random_scale(y)

for res in [result_1, result_2, result_3, result_4]:
  assert(res.shape == (2,2))
  assert(res.dtype == tf.float32)
  assert(np.max(np.array(res)) <= 1.0)
  assert(np.min(np.array(res)) >= 0.0)

print("Noise functions look good!")

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
            filters=3,
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
    def call(self, images):
      encoded = self.encoder(images)
      decoded = self.decoder(encoded)
      return decoded

    @tf.function
    def loss_function(self, encoded, originals):
      encoded = tf.dtypes.cast(encoded, tf.float32)
      originals = tf.dtypes.cast(originals, tf.float32)
      return tf.reduce_sum(tf.square(originals - encoded))
    
def train(model, optimizer, images, noise_function):
  corrupted = noise_function(images)
  uncorrupted = images

  with tf.GradientTape() as tape:
    predictions = model(corrupted)
    loss = model.loss_function(predictions, uncorrupted)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def total_loss(model, images, noise_function):
  corrupted = noise_function(images)
  uncorrupted = images
  predictions = model(corrupted)
  sum_loss = model.loss_function(predictions, uncorrupted)
  return sum_loss



# This code is helper code to plot the cifar10 images so you
# can see your autoencoder in action!
# %matplotlib inline
import matplotlib.pyplot as plt

n_examples = 10
batch_size = 100
n_epochs = 25


(train_images, _), (test_images, _) = tf.keras.datasets.cifar10.load_data()
test_images = test_images / 255
example_images = test_images[:n_examples]
train_images = train_images / 255


def showImages(model, noise_function):
  corrupted = noise_function(example_images)
  recon = tf.clip_by_value(model(corrupted), 0, 1)

  fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
  for example_i in range(n_examples):
      axs[0][example_i].imshow(corrupted[example_i])
      axs[1][example_i].imshow(recon[example_i])
  plt.show()


# Runs the autoencoder
# We'll just be testing the random_noise function for training
for noise_function in [random_noise]:
  print("Showing autoencoder for noise function: {0}".format(noise_function))
  model = AutoEncoder()
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
  for i in range(n_epochs):
    for j in range(0, len(train_images), batch_size):
      train(model, optimizer, train_images[j:j+batch_size], noise_function)

    print("Epoch: ", i)
    sum_loss = total_loss(model, test_images, noise_function)
    print("Total Loss: {0}".format(sum_loss))
    showImages(model, noise_function)