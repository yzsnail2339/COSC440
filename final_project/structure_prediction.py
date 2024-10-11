import os
# suppress silly log messages from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import random
import structure_prediction_utils as utils
from Autoencoder import AutoEncoder
import res_autoencoder as res
import vit_model as vit
from tensorflow import keras
import csv
import time

avg_loss_list_epochs = []
avg_mse_loss_list_epochs = []
validate_loss_list_epochs = []
test_loss_epochs = []


class ProteinStructurePredictor0(keras.Model):
    def __init__(self):
        super().__init__()
        self.layer0 = keras.layers.Conv2D(5, 5, activation='gelu', padding="same")
        self.layer1 = keras.layers.Conv2D(1, 1, activation='gelu', padding="same")

    #@tf.function
    def call(self, inputs, mask=None):
        primary_one_hot = inputs

        # outer sum to get a NUM_RESIDUES x NUM_RESIDUES x embedding size
        x = tf.expand_dims(inputs, -2) + tf.expand_dims(inputs, -3)   

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
        primary_one_hot = inputs

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
        primary_one_hot = inputs
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
        primary_one_hot = inputs
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
        self.layer0 = keras.layers.Conv2D(7, 5, activation='gelu', padding="same")
        # self.layer1 = keras.layers.Conv2D(1, 1, activation='gelu', padding="same")
        self.attention = keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)
        self.dense1 = keras.layers.Dense(64, activation='gelu')
        # self.dense2 = keras.layers.Dense(10, activation='gelu')
        self.resnet = res.resnet34()
        self.add = keras.layers.Add()
    #@tf.function
    def call(self, inputs, mask=None):
        primary_one_hot = inputs
        attention_output = self.attention(primary_one_hot, primary_one_hot, primary_one_hot)
        # x = primary_one_hot + attention_output
        x = self.add([primary_one_hot , attention_output])
        x = self.dense1(x)
        # outer sum to get a NUM_RESIDUES x NUM_RESIDUES x embedding size
        x = self.add([tf.expand_dims(x, -2) , tf.expand_dims(x, -3)])
        # x = tf.expand_dims(x, -2) + tf.expand_dims(x, -3)
        # filter the initial representation into an embedded representation
        x = self.layer0(x)

        # add positional distance information
        r = tf.range(0, utils.NUM_RESIDUES, dtype=tf.float32)
        distances = tf.abs(tf.expand_dims(r, -1) - tf.expand_dims(r, -2))
        distances_bc = tf.expand_dims(
            tf.broadcast_to(distances, [tf.shape(primary_one_hot)[0], utils.NUM_RESIDUES, utils.NUM_RESIDUES]), -1)
        distances_bc = distances_bc * tf.expand_dims(mask, axis=-1)


        x = tf.concat([x, x * distances_bc, distances_bc], axis=-1)
        x = self.resnet(x)
        x = x * tf.expand_dims(mask, axis=-1)
        # x = self.dense2(x)
        # x = self.layer1(x)
        return x


class ProteinStructurePredictor6(keras.Model):
    def __init__(self):
        super().__init__()
        self.layer0 = keras.layers.Conv2D(7, 5, activation='gelu', padding="same")
        # self.layer1 = keras.layers.Conv2D(1, 1, activation='gelu', padding="same")
        self.attention = keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)
        self.dense1 = keras.layers.Dense(64, activation='gelu')
        # self.dense2 = keras.layers.Dense(10, activation='gelu')
        self.resnet = res.resnet34()
        self.vit = vit.vit_base_patch16_224_in21k()
        self.add = keras.layers.Add()
    #@tf.function
    def call(self, inputs, mask=None):
        primary_one_hot = inputs
        attention_output = self.attention(primary_one_hot, primary_one_hot, primary_one_hot)
        x = self.add([primary_one_hot , attention_output])
        x = self.dense1(x)
        # outer sum to get a NUM_RESIDUES x NUM_RESIDUES x embedding size
        x = self.add([tf.expand_dims(x, -2) , tf.expand_dims(x, -3)])
        # filter the initial representation into an embedded representation
        x = self.layer0(x)

        # add positional distance information
        r = tf.range(0, utils.NUM_RESIDUES, dtype=tf.float32)
        distances = tf.abs(tf.expand_dims(r, -1) - tf.expand_dims(r, -2))
        distances_bc = tf.expand_dims(
            tf.broadcast_to(distances, [tf.shape(primary_one_hot)[0], utils.NUM_RESIDUES, utils.NUM_RESIDUES]), -1)
        distances_bc = distances_bc * tf.expand_dims(mask, axis=-1)


        x = tf.concat([x, x * distances_bc, distances_bc], axis=-1)
        x = self.resnet(x)
        # x = x * tf.expand_dims(mask, axis=-1)
        x = self.vit(x)
        # x = self.dense2(x)
        # x = self.layer1(x)
        return x
  
def get_n_records(batch):
    return batch['primary_onehot'].shape[0]
def get_input_output_masks(batch):
    inputs = batch['primary_onehot']
    outputs = batch['true_distances']
    masks = batch['distance_mask']
    return inputs, outputs, masks

def print_mean_batch_loss(mean_avg_loss_list,mean_avg_mse_loss_list,mean_validate_loss_list):
    print("mean of batch loss")
    print(f'mean_avg_loss_list : {mean_avg_loss_list}')
    print(f'mean_avg_mse_loss_list : {mean_avg_mse_loss_list}')
    print(f'mean_validate_loss_list : {mean_validate_loss_list}')

def record_each_epochs(avg_loss_list,avg_mse_loss_list,validate_loss_list,time_batch):
    mean_avg_loss_list = tf.reduce_sum(avg_loss_list)/time_batch
    mean_avg_mse_loss_list = tf.reduce_sum(avg_mse_loss_list)/time_batch
    mean_validate_loss_list = tf.reduce_sum(validate_loss_list)/time_batch
    print_mean_batch_loss(mean_avg_loss_list,mean_avg_mse_loss_list,mean_validate_loss_list)
    avg_loss_list_epochs.append(mean_avg_loss_list)
    avg_mse_loss_list_epochs.append(mean_avg_mse_loss_list)
    validate_loss_list_epochs.append(mean_validate_loss_list)


def save_to_file(avg_loss_list_epochs, avg_mse_loss_list_epochs, validate_loss_list_epochs):
    # 创建新的文件夹以存储损失记录
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    folder_name = 'each_epochs_loss'
    os.makedirs(folder_name, exist_ok=True)

    # 创建文件路径
    file_path = os.path.join(folder_name, f'each_epochs_loss_{timestamp}.csv')

    # 将损失值保存到 CSV 文件中
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(['Epoch', 'Avg Loss', 'Avg MSE Loss', 'Validation Loss'])
        
        # 写入每一轮的损失值
        for i in range(len(avg_loss_list_epochs)):
            writer.writerow([i + 1, avg_loss_list_epochs[i], avg_mse_loss_list_epochs[i], validate_loss_list_epochs[i]])

    print(f"Loss records saved to '{file_path}'")


def train(model, train_dataset, validate_dataset=None, train_loss=utils.mse_point_loss):
    '''
    Trains the model
    '''
    time_batch = 0
    avg_loss = 0.
    avg_mse_loss = 0.
    avg_loss_list = []
    avg_mse_loss_list = []
    validate_loss_list = []

    def print_loss():
        if validate_dataset is not None:
            validate_loss = 0.

            validate_batches = 0.
            #be care of the validate_dataset.batch(1024)
            for batch in validate_dataset.batch(model.batch_size):
                validate_inputs, validate_outputs, validate_masks = get_input_output_masks(batch)
                validate_preds = model.call(validate_inputs, validate_masks)

                validate_loss += tf.reduce_sum(train_loss(validate_preds, validate_outputs, validate_masks)) / get_n_records(batch)
                validate_batches += 1
            validate_loss /= validate_batches
        else:
            validate_loss = float('NaN')
        avg_loss_list.append(avg_loss)
        avg_mse_loss_list.append(avg_mse_loss)
        validate_loss_list.append(validate_loss)
        print(
            f'train loss {avg_loss:.3f} train mse loss {avg_mse_loss:.3f} validate mse loss {validate_loss:.3f}')
        # print(f'train loss {avg_loss:.3f} train mse loss {avg_mse_loss:.3f}')
    first = True
    for batch in train_dataset:
        time_batch += 1
        inputs, labels, masks = get_input_output_masks(batch)
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            outputs = model(inputs, masks)

            l = train_loss(outputs, labels, masks)
            batch_loss = tf.reduce_sum(l)
        gradients = tape.gradient(batch_loss, model.trainable_weights)
        avg_loss = batch_loss / get_n_records(batch)
        avg_mse_loss = tf.reduce_sum(utils.mse_loss(outputs, labels, masks)) / get_n_records(batch) 
        # avg_mse_loss = tf.reduce_mean(avg_mse_loss_list)
        
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)  #梯度裁剪
        model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        print_loss()
        if first:
            model.summary()
            # model.resnet.summary()
            # model.resnet.get_layer('block1').summary()
            first = False
    record_each_epochs(avg_loss_list,avg_mse_loss_list,validate_loss_list,time_batch)
    utils.display_three_loss(avg_loss_list, avg_mse_loss_list, validate_loss_list)

def test(model, test_records, viz=False):
    test_loss_mean = 0
    time_batch = 0
    for batch in test_records.batch(model.batch_size):
        time_batch += 1
        test_inputs, test_outputs, test_masks = get_input_output_masks(batch)
        test_preds = model.call(test_inputs, test_masks)
        test_loss = tf.reduce_sum(utils.mse_point_loss(test_preds, test_outputs, test_masks)) / get_n_records(batch)
        print(f'test mse loss {test_loss:.3f}')
        test_loss_mean += test_loss
    
    if viz:
        model.summary()
        r = random.randint(0, test_preds.shape[0] -1)
        utils.display_two_structures(test_preds[r], test_outputs[r], test_masks[r])
        viz = False
    test_loss_epochs.append(test_loss_mean)


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
    model = ProteinStructurePredictor6()
    model.optimizer = keras.optimizers.Adam(learning_rate=1e-2)
    model.batch_size = 8 #1024



    epochs = 5
    # Iterate over epochs.
    for epoch in range(epochs):
        epoch_training_records = training_records.shuffle(buffer_size=256).batch(model.batch_size, drop_remainder=False)
        print("Start of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        # strategy.run(train, args=(model, epoch_training_records, validate_records))
        train(model, epoch_training_records, validate_records)
        # strategy.run(test, args=(model, test_records, True))
        test(model, test_records, True)

    print("----------each epoch loss----------")
    utils.display_three_loss(avg_loss_list_epochs, avg_mse_loss_list_epochs, validate_loss_list_epochs)
    test(model, test_records, True)
    utils.display_test_loss_epochs(test_loss_epochs)
    save_to_file(avg_loss_list_epochs, avg_mse_loss_list_epochs, validate_loss_list_epochs)
    model.save(data_folder + 'model.keras')


if __name__ == '__main__':
    data_folder = './final_project/'


    main(data_folder)