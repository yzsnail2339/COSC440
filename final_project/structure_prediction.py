import os
# suppress silly log messages from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import random
import structure_prediction_utils as utils
from Autoencoder import AutoEncoder
import res_autoencoder as res
from tensorflow import keras

class ProteinStructurePredictor0(keras.Model):
    def __init__(self):
        super().__init__()
        self.layer0 = keras.layers.Conv2D(5, 5, activation='gelu', padding="same")
        self.layer1 = keras.layers.Conv2D(1, 1, activation='gelu', padding="same")

    #@tf.function
    def call(self, inputs, mask=None):
        primary_one_hot = inputs['primary_onehot']

        # outer sum to get a NUM_RESIDUES x NUM_RESIDUES x embedding size
        x = tf.expand_dims(primary_one_hot, -2) + tf.expand_dims(primary_one_hot, -3)   

        # filter the initial representation into an embedded representation
        x = self.layer0(x)


        # add positional distance information
        r = tf.range(0, utils.NUM_RESIDUES, dtype=tf.float32)
        distances = tf.abs(tf.expand_dims(r, -1) - tf.expand_dims(r, -2))
        distances_bc = tf.expand_dims(
            tf.broadcast_to(distances, [tf.shape(primary_one_hot)[0], utils.NUM_RESIDUES, utils.NUM_RESIDUES]), -1)

        x = tf.concat([x, x * distances_bc, distances_bc], axis=-1)
        # x = distances_bc
        # generate result
        x = self.layer1(x)

        return x
    
class ProteinStructurePredictor1(keras.Model):
    def __init__(self):
        super().__init__()
        self.layer0 = keras.layers.Conv2D(5, 5, activation='gelu', padding="same")
        self.layer1 = keras.layers.Conv2D(1, 1, activation='gelu', padding="same")
        self.pooling = keras.layers.MaxPooling2D(pool_size=(8, 8))
        self.upsampling = keras.layers.UpSampling2D(size=(8, 8))
        self.attention = keras.layers.MultiHeadAttention(num_heads=1, key_dim=2)
        self.dense = keras.layers.Dense(64, activation='gelu')

    #@tf.function
    def call(self, inputs, mask=None):
        primary_one_hot = inputs['primary_onehot']

        # outer sum to get a NUM_RESIDUES x NUM_RESIDUES x embedding size
        x = tf.expand_dims(primary_one_hot, -2) + tf.expand_dims(primary_one_hot, -3)   

        # filter the initial representation into an embedded representation
        x = self.layer0(x)


        # add positional distance information
        r = tf.range(0, utils.NUM_RESIDUES, dtype=tf.float32)
        distances = tf.abs(tf.expand_dims(r, -1) - tf.expand_dims(r, -2))
        distances_bc = tf.expand_dims(
            tf.broadcast_to(distances, [tf.shape(primary_one_hot)[0], utils.NUM_RESIDUES, utils.NUM_RESIDUES]), -1)

        x = tf.concat([x, x * distances_bc, distances_bc], axis=-1)
        x = self.pooling(x)
        attention_output = self.attention(x, x, x)
        x = x + attention_output
        x = self.dense(x)
        # x = distances_bc
        x = self.upsampling(x)
        # generate result
        x = self.layer1(x)
        return x

class ProteinStructurePredictor2(keras.Model):
    def __init__(self):
        super().__init__()
        self.layer0 = keras.layers.Conv2D(5, 5, activation='gelu', padding="same")
        self.layer1 = keras.layers.Conv2D(1, 1, activation='gelu', padding="same")
        self.attention = keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)
        self.dense = keras.layers.Dense(32, activation='gelu')

    @tf.function
    def call(self, inputs, mask=None):
        primary_one_hot = inputs['primary_onehot']
        attention_output = self.attention(primary_one_hot, primary_one_hot, primary_one_hot)
        x = primary_one_hot + attention_output
        x = self.dense(x)
        # outer sum to get a NUM_RESIDUES x NUM_RESIDUES x embedding size
        x = tf.expand_dims(x, -2) + tf.expand_dims(x, -3)   
        # filter the initial representation into an embedded representation
        x = self.layer0(x)


        # add positional distance information
        r = tf.range(0, utils.NUM_RESIDUES, dtype=tf.float32)
        distances = tf.abs(tf.expand_dims(r, -1) - tf.expand_dims(r, -2))
        distances_bc = tf.expand_dims(
            tf.broadcast_to(distances, [tf.shape(primary_one_hot)[0], utils.NUM_RESIDUES, utils.NUM_RESIDUES]), -1)

        x = tf.concat([x, x * distances_bc, distances_bc], axis=-1)
        # x = distances_bc
        # generate result
        x = self.layer1(x)
        return x
    
class ProteinStructurePredictor3(keras.Model):
    def __init__(self):
        super().__init__()
        self.layer0 = keras.layers.Conv2D(5, 5, activation='gelu', padding="same")
        self.layer1 = keras.layers.Conv2D(1, 1, activation='gelu', padding="same")
        self.pooling = keras.layers.MaxPooling2D(pool_size=(8, 8))
        self.upsampling = keras.layers.UpSampling2D(size=(8, 8))
        self.attention = keras.layers.MultiHeadAttention(num_heads=1, key_dim=2)
        self.dense = keras.layers.Dense(64, activation='gelu')
        self.autencode = AutoEncoder()

    #@tf.function
    def call(self, inputs, mask=None):
        primary_one_hot = inputs['primary_onehot']
        attention_output = self.attention(primary_one_hot, primary_one_hot, primary_one_hot)
        x = primary_one_hot + attention_output
        # outer sum to get a NUM_RESIDUES x NUM_RESIDUES x embedding size
        x = tf.expand_dims(x, -2) + tf.expand_dims(x, -3)   

        # filter the initial representation into an embedded representation
        x = self.layer0(x)


        # add positional distance information
        r = tf.range(0, utils.NUM_RESIDUES, dtype=tf.float32)
        distances = tf.abs(tf.expand_dims(r, -1) - tf.expand_dims(r, -2))
        distances_bc = tf.expand_dims(
            tf.broadcast_to(distances, [tf.shape(primary_one_hot)[0], utils.NUM_RESIDUES, utils.NUM_RESIDUES]), -1)

        x = tf.concat([x, x * distances_bc, distances_bc], axis=-1)
        x = self.pooling(x)
        x = self.dense(x)
        x = self.autencode(x)
        # x = distances_bc
        x = self.upsampling(x)
        # generate result
        x = self.layer1(x)
        return x
    
class ProteinStructurePredictor5(keras.Model):
    def __init__(self):
        super().__init__()
        self.layer0 = keras.layers.Conv2D(5, 5, activation='gelu', padding="same")
        self.layer1 = keras.layers.Conv2D(1, 1, activation='gelu', padding="same")
        self.attention = keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)
        self.dense = keras.layers.Dense(64, activation='gelu')
        self.resnet = res.resnet50()

    #@tf.function
    def call(self, inputs, mask=None):
        primary_one_hot = inputs['primary_onehot']
        attention_output = self.attention(primary_one_hot, primary_one_hot, primary_one_hot)
        x = primary_one_hot + attention_output
        x = self.dense(x)
        # outer sum to get a NUM_RESIDUES x NUM_RESIDUES x embedding size
        x = tf.expand_dims(x, -2) + tf.expand_dims(x, -3)
        # filter the initial representation into an embedded representation
        x = self.layer0(x)
        # add positional distance information
        r = tf.range(0, utils.NUM_RESIDUES, dtype=tf.float32)
        distances = tf.abs(tf.expand_dims(r, -1) - tf.expand_dims(r, -2))
        distances_bc = tf.expand_dims(
            tf.broadcast_to(distances, [tf.shape(primary_one_hot)[0], utils.NUM_RESIDUES, utils.NUM_RESIDUES]), -1)
        x = tf.concat([x, x * distances_bc, distances_bc], axis=-1)
        x = self.resnet(x)
        x = self.layer1(x)
        return x

def get_n_records(batch):
    return batch['primary_onehot'].shape[0]
def get_input_output_masks(batch):
    inputs = {'primary_onehot':batch['primary_onehot']}
    outputs = batch['true_distances']
    masks = batch['distance_mask']
    return inputs, outputs, masks
    
def train(model, train_dataset, validate_dataset=None, train_loss=utils.mse_loss):
    '''
    Trains the model
    '''

    avg_loss = 0.
    avg_mse_loss = 0.
    avg_loss_list = []
    avg_mse_loss_list = []

    def print_loss():
        if validate_dataset is not None:
            validate_loss = 0.

            validate_batches = 0.
            #be care of the validate_dataset.batch(1024)
            for batch in validate_dataset.batch(model.batch_size):
                validate_inputs, validate_outputs, validate_masks = get_input_output_masks(batch)
                validate_preds = model.call(validate_inputs, validate_masks)

                validate_loss += tf.reduce_sum(utils.mse_loss(validate_preds, validate_outputs, validate_masks)) / get_n_records(batch)
                validate_batches += 1
            validate_loss /= validate_batches
        else:
            validate_loss = float('NaN')
        print(
            f'train loss {avg_loss:.3f} train mse loss {avg_mse_loss:.3f} validate mse loss {validate_loss:.3f}')
        # print(f'train loss {avg_loss:.3f} train mse loss {avg_mse_loss:.3f}')
    first = True
    for batch in train_dataset:
        inputs, labels, masks = get_input_output_masks(batch)
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            outputs = model(inputs, masks)

            l = train_loss(outputs, labels, masks)
            batch_loss = tf.reduce_sum(l)
            gradients = tape.gradient(batch_loss, model.trainable_weights)
            avg_loss = batch_loss / get_n_records(batch)
            avg_mse_loss = tf.reduce_sum(utils.mse_loss(outputs, labels, masks)) / get_n_records(batch)

        model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        avg_loss_list.append(avg_loss)
        avg_mse_loss_list.append(avg_mse_loss)
        print_loss()
        if first:
            print(model.summary())
            print(model.resnet.summary()) 
            print(model.resnet.get_layer('block1').summary())
            first = False
    utils.display_two_loss(avg_loss_list, avg_mse_loss_list)

def test(model, test_records, viz=False):
    for batch in test_records.batch(model.batch_size):
        test_inputs, test_outputs, test_masks = get_input_output_masks(batch)
        test_preds = model.call(test_inputs, test_masks)
        test_loss = tf.reduce_sum(utils.mse_loss(test_preds, test_outputs, test_masks)) / get_n_records(batch)
        print(f'test mse loss {test_loss:.3f}')

    if viz:
        print(model.summary())
        r = random.randint(0, test_preds.shape[0])
        utils.display_two_structures(test_preds[r], test_outputs[r], test_masks[r])
        viz = False


# 递归显示整个主模型的结构

def main(data_folder):
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     for gpu in gpus:
    #         tf.config.experimental.set_memory_growth(gpu, True)

    # tf.debugging.set_log_device_placement(True)

    training_records = utils.load_preprocessed_data(data_folder, 'training.tfr')
    validate_records = utils.load_preprocessed_data(data_folder, 'validation.tfr')
    test_records = utils.load_preprocessed_data(data_folder, 'testing.tfr')


    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    model = ProteinStructurePredictor5()
    model.optimizer = keras.optimizers.Adam(learning_rate=1e-2)
    model.batch_size = 32   #1024




    epochs = 1
    # Iterate over epochs.
    for epoch in range(epochs):
        epoch_training_records = training_records.shuffle(buffer_size=256).batch(model.batch_size, drop_remainder=False)
        print("Start of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        # strategy.run(train, args=(model, epoch_training_records, validate_records))
        train(model, epoch_training_records, validate_records)
        # strategy.run(test, args=(model, test_records, True))
        test(model, test_records, True)


    test(model, test_records, True)

    model.save(data_folder + '/model')


if __name__ == '__main__':
    data_folder = './final_project/'


    main(data_folder)