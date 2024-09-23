import os
# suppress silly log messages from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import random
import structure_prediction_utils as utils
from tensorflow import keras

class ProteinStructurePredictor0(keras.Model):
    def __init__(self):
        super().__init__()
        # self.layer0 = keras.layers.Conv2D(5, 5, activation='gelu', padding="same")
        self.layer1 = keras.layers.Conv2D(1, 1, activation='gelu', padding="same")

    #@tf.function
    def call(self, inputs, mask=None):
        primary_one_hot = inputs['primary_onehot']

        # outer sum to get a NUM_RESIDUES x NUM_RESIDUES x embedding size
        # x = tf.expand_dims(primary_one_hot, -2) + tf.expand_dims(primary_one_hot, -3)

        # filter the initial representation into an embedded representation
        # x = self.layer0(x)


        # add positional distance information
        r = tf.range(0, utils.NUM_RESIDUES, dtype=tf.float32)
        distances = tf.abs(tf.expand_dims(r, -1) - tf.expand_dims(r, -2))
        distances_bc = tf.expand_dims(
            tf.broadcast_to(distances, [primary_one_hot.shape[0], utils.NUM_RESIDUES, utils.NUM_RESIDUES]), -1)

        # x = tf.concat([x, x * distances_bc, distances_bc], axis=-1)
        x = distances_bc
        # generate result
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

    def print_loss():
        if validate_dataset is not None:
            validate_loss = 0.

            validate_batches = 0.
            for batch in validate_dataset.batch(1024):
                validate_inputs, validate_outputs, validate_masks = get_input_output_masks(batch)
                validate_preds = model.call(validate_inputs, validate_masks)

                validate_loss += tf.reduce_sum(utils.mse_loss(validate_preds, validate_outputs, validate_masks)) / get_n_records(batch)
                validate_batches += 1
            validate_loss /= validate_batches
        else:
            validate_loss = float('NaN')
        print(
            f'train loss {avg_loss:.3f} train mse loss {avg_mse_loss:.3f} validate mse loss {validate_loss:.3f}')
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

        print_loss()

        if first:
            print(model.summary())
            first = False

def test(model, test_records, viz=False):
    for batch in test_records.batch(1024):
        test_inputs, test_outputs, test_masks = get_input_output_masks(batch)
        test_preds = model.call(test_inputs, test_masks)
        test_loss = tf.reduce_sum(utils.mse_loss(test_preds, test_outputs, test_masks)) / get_n_records(batch)
        print(f'test mse loss {test_loss:.3f}')

    if viz:
        print(model.summary())
        r = random.randint(0, test_preds.shape[0])
        utils.display_two_structures(test_preds[r], test_outputs[r], test_masks[r])

def main(data_folder):
    training_records = utils.load_preprocessed_data(data_folder, 'training.tfr')
    validate_records = utils.load_preprocessed_data(data_folder, 'validation.tfr')
    test_records = utils.load_preprocessed_data(data_folder, 'testing.tfr')

    model = ProteinStructurePredictor0()
    model.optimizer = keras.optimizers.Adam(learning_rate=1e-2)
    model.batch_size = 1024
    epochs = 5
    # Iterate over epochs.
    for epoch in range(epochs):
        epoch_training_records = training_records.shuffle(buffer_size=256).batch(model.batch_size, drop_remainder=False)
        print("Start of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        train(model, epoch_training_records, validate_records)

        test(model, test_records, True)

    test(model, test_records, True)

    model.save(data_folder + '/model')


if __name__ == '__main__':
    data_folder = 'd:/UC_Master/COSC440/assignment_all/final_project/'


    main(data_folder)