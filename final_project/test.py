import tensorflow as tf
from tensorflow.keras import layers

# 定义输入张量
input1 = tf.keras.Input(shape=(32, 32, 64))
input2 = tf.keras.Input(shape=(32, 32, 64))

# 使用 Add 层将两个输入相加
added = layers.Add()([input1, input2])

# 创建模型
model = tf.keras.Model(inputs=[input1, input2], outputs=added)

model.summary()
