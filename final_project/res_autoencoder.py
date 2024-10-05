from tensorflow.keras import layers, Model, Sequential
import tensorflow as tf

class BasicBlock(layers.Layer):
    expansion = 1

    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(out_channel, kernel_size=3, strides=strides,
                                   padding="same", use_bias=False)
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        # -----------------------------------------
        self.conv2 = layers.Conv2D(out_channel, kernel_size=3, strides=1,
                                   padding="same", use_bias=False)
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        # -----------------------------------------
        self.downsample = downsample
        self.relu = layers.ReLU()
        self.add = layers.Add()

    def call(self, inputs, training=True):
        identity = inputs
        if self.downsample is not None:
            identity = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = self.add([identity, x])
        x = self.relu(x)

        return x
    
# class Decoder(tf.keras.layers.Layer):
#     def __init__(self):
#         super(Decoder, self).__init__()
#         self.decoder_deconv_1 = tf.keras.layers.Conv2DTranspose(
#             filters=128,
#             kernel_size=3,
#             strides=2,
#             padding='same',
#             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
#             activation=tf.keras.layers.LeakyReLU(alpha=0.2),
#             name="deconv1"
#         )
#         self.decoder_deconv_2 = tf.keras.layers.Conv2DTranspose(
#             filters=64,
#             kernel_size=3,
#             strides=2,
#             padding='same',
#             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
#             activation=tf.keras.layers.LeakyReLU(alpha=0.2),
#             name="deconv2"
#         )
#         self.decoder_deconv_3 = tf.keras.layers.Conv2DTranspose(
#             filters=32,
#             kernel_size=3,
#             strides=2,
#             padding='same',
#             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
#             activation=tf.keras.layers.LeakyReLU(alpha=0.2),
#             name="deconv3"
#         )
#         self.decoder_deconv_4 = tf.keras.layers.Conv2DTranspose(
#             filters=16,
#             kernel_size=3,
#             strides=2,
#             padding='same',
#             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
#             activation=tf.keras.layers.LeakyReLU(alpha=0.2),
#             name="deconv4"
#         )
#         self.decoder_deconv_5 = tf.keras.layers.Conv2DTranspose(
#             filters=1,
#             kernel_size=3,
#             strides=2,
#             padding='same',
#             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
#             activation=tf.keras.layers.LeakyReLU(alpha=0.2),
#             name="deconv5"
#         )

#     def call(self, encoder_output):
#       x = self.decoder_deconv_1(encoder_output)
#       x = self.decoder_deconv_2(x)
#       x = self.decoder_deconv_3(x)
#       x = self.decoder_deconv_4(x)
#       x = self.decoder_deconv_5(x)
#       return x

class Decoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder_deconv_1 = tf.keras.layers.Conv2DTranspose(
            filters=256,
            kernel_size=3,
            strides=2,
            padding='same',
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            use_bias=False, 
            name="deconv1"
        )
        self.decoder_bn1 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.decoder_activation1 = tf.keras.layers.LeakyReLU(alpha=0.2)

        self.decoder_deconv_2 = tf.keras.layers.Conv2DTranspose(
            filters=128,
            kernel_size=3,
            strides=2,
            padding='same',
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            use_bias=False,
            name="deconv2"
        )
        self.decoder_bn2 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.decoder_activation2 = tf.keras.layers.LeakyReLU(alpha=0.2)

        self.decoder_deconv_3 = tf.keras.layers.Conv2DTranspose(
            filters=64,
            kernel_size=3,
            strides=2,
            padding='same',
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            use_bias=False,
            name="deconv3"
        )
        self.decoder_bn3 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.decoder_activation3 = tf.keras.layers.LeakyReLU(alpha=0.2)

        self.decoder_deconv_4 = tf.keras.layers.Conv2DTranspose(
            filters=32,
            kernel_size=3,
            strides=2,
            padding='same',
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            use_bias=False,
            name="deconv4"
        )
        self.decoder_bn4 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.decoder_activation4 = tf.keras.layers.LeakyReLU(alpha=0.2)

        self.decoder_deconv_5 = tf.keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=3,
            strides=2,
            padding='same',
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            use_bias=False,
            name="deconv5"
        )
        self.decoder_activation5 = tf.keras.layers.LeakyReLU(alpha=0.2)

    def call(self, encoder_output, training=True):
        x = self.decoder_deconv_1(encoder_output)
        x = self.decoder_bn1(x, training=training)
        x = self.decoder_activation1(x)

        x = self.decoder_deconv_2(x)
        x = self.decoder_bn2(x, training=training)
        x = self.decoder_activation2(x)

        x = self.decoder_deconv_3(x)
        x = self.decoder_bn3(x, training=training)
        x = self.decoder_activation3(x)

        x = self.decoder_deconv_4(x)
        x = self.decoder_bn4(x, training=training)
        x = self.decoder_activation4(x)

        x = self.decoder_deconv_5(x)
        x = self.decoder_activation5(x)

        return x

class Bottleneck(layers.Layer):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(out_channel, kernel_size=1, use_bias=False, name="conv1")
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1_BatchNorm")
        # -----------------------------------------
        self.conv2 = layers.Conv2D(out_channel, kernel_size=3, use_bias=False,
                                   strides=strides, padding="same", name="conv2")
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv2_BatchNorm")
        # -----------------------------------------
        self.conv3 = layers.Conv2D(out_channel * self.expansion, kernel_size=1, use_bias=False, name="conv3")
        self.bn3 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv3_BatchNorm")
        # -----------------------------------------
        self.relu = layers.ReLU()
        self.downsample = downsample
        self.add = layers.Add()

    def call(self, inputs, training=True):
        identity = inputs
        if self.downsample is not None:
            identity = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.add([x, identity])
        x = self.relu(x)

        return x


def _make_layer(block, in_channel, channel, block_num, name, strides=1):
    downsample = None
    if strides != 1 or in_channel != channel * block.expansion:
        downsample = Sequential([
            layers.Conv2D(channel * block.expansion, kernel_size=1, strides=strides,
                          use_bias=False, name="conv1_d"),
            layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5, name="BatchNorm_d")
        ], name="shortcut")

    layers_list = []
    layers_list.append(block(channel, downsample=downsample, strides=strides, name="unit_1"))

    for index in range(1, block_num):
        layers_list.append(block(channel, name="unit_" + str(index + 1)))

    return Sequential(layers_list, name=name)


def _resnet(block, blocks_num, im_width=256, im_height=256):
    # tensorflow中的tensor通道排序是NHWC
    # (None, 224, 224, 3)
    # (None, 256, 256, 21)
    input_image = layers.Input(shape=(im_height, im_width, 15), dtype="float32")
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2,
                      padding="same", use_bias=False, name="conv1")(input_image)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1_BatchNorm")(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)

    # x = _make_layer(block, x.shape[-1], 32, blocks_num[0], name="block1")(x)
    # x = _make_layer(block, x.shape[-1], 64, blocks_num[1], strides=2, name="block2")(x)
    # x = _make_layer(block, x.shape[-1], 128, blocks_num[2], strides=2, name="block3")(x)
    # x = _make_layer(block, x.shape[-1], 256, blocks_num[3], strides=2, name="block4")(x)
    x = _make_layer(block, x.shape[-1], 64, blocks_num[0], name="block1")(x)
    x = _make_layer(block, x.shape[-1], 128, blocks_num[1], strides=2, name="block2")(x)
    x = _make_layer(block, x.shape[-1], 256, blocks_num[2], strides=2, name="block3")(x)
    x = _make_layer(block, x.shape[-1], 512, blocks_num[3], strides=2, name="block4")(x)

    predict = Decoder()(x)


    model = Model(inputs=input_image, outputs=predict, name="ResNet_autoencoder")

    return model


def resnet34(im_width=256, im_height=256):
    return _resnet(BasicBlock, [3, 4, 6, 3], im_width, im_height)


def resnet50(im_width=256, im_height=256):
    return _resnet(Bottleneck, [3, 4, 6, 3], im_width, im_height)


def resnet101(im_width=224, im_height=224):
    return _resnet(Bottleneck, [3, 4, 23, 3], im_width, im_height)

