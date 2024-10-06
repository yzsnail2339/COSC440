import glob, os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.io import FixedLenFeature, FixedLenSequenceFeature

NUM_RESIDUES = 256
NUM_EXTRA_SEQ = 10
NUM_DIMENSIONS = 3
NUM_AMINO_ACIDS = 21
#Abbreviation	1 letter abbreviation	Amino acid name
#twenty amino acids (that make up proteins) + 1 for representing a gap in an alignment
AMINO_ACIDS = [record.split() for record in
"""Ala	A	Alanine
Arg	R	Arginine
Asn	N	Asparagine
Asp	D	Aspartic acid
Cys	C	Cysteine
Gln	Q	Glutamine
Glu	E	Glutamic acid
Gly	G	Glycine
His	H	Histidine
Ile	I	Isoleucine
Leu	L	Leucine
Lys	K	Lysine
Met	M	Methionine
Phe	F	Phenylalanine
Pro	P	Proline
Ser	S	Serine
Thr	T	Threonine
Trp	W	Tryptophan
Tyr	Y	Tyrosine
Val	V	Valine
GAP _   Alignment gap""".splitlines()]
# unused codes
# Pyl	O	Pyrrolysine
# Sec	U	Selenocysteine
# Asx	B	Aspartic acid or Asparagine
# Glx	Z	Glutamic acid or Glutamine
# Xaa	X	Any amino acid
# Xle	J	Leucine or Isoleucine
AMINO_ACID_MAP = {r[1]:r[2] for r in AMINO_ACIDS}
AMINO_ACID_BY_INDEX = list(sorted(AMINO_ACID_MAP.keys())) #0-20 inclusive, protein net numbering
AMINO_ACID_NUMBER = {key:number for number, key in enumerate(AMINO_ACID_BY_INDEX)}
AMINO_ACID_GAP_INDEX = AMINO_ACID_BY_INDEX.index('_')
def to_amino_acid_string(amino_acid_indices):
    return "".join([AMINO_ACID_BY_INDEX[x] for x in amino_acid_indices])
def to_amino_acid_numerical(amino_acid_codes):
    return np.array([AMINO_ACID_NUMBER[x] if x != '-' else AMINO_ACID_GAP_INDEX for x in amino_acid_codes], dtype=np.int64)
ID_MAXLEN = 16


def mse_loss(predicted_distance_matrix, true_distance_matrix, mask):
    diff = (tf.squeeze(predicted_distance_matrix)-true_distance_matrix) ** 2
    diff *= mask
    return tf.reduce_sum(diff, axis=(-1, -2))

def mse_point_loss(predicted_distance_matrix, true_distance_matrix, mask):
    diff = tf.abs(tf.squeeze(predicted_distance_matrix)-true_distance_matrix)
    diff *= mask
    return tf.reduce_sum(diff, axis=(-1, -2)) / tf.reduce_sum(mask)

def per_residue_distance(points):
    return tf.sqrt(1e-10 + tf.reduce_sum((tf.expand_dims(points, -2)-
        tf.expand_dims(points, -3)) ** 2, axis=-1))
    # return tf.sqrt(1e-10 + tf.reduce_sum(
    #     (points[:, :, None] - points[:, None, :]) ** 2, axis=-1))


def decode_preprocessed_fn(example_proto, debug=False):
    feature_description = {
        'id': tf.io.FixedLenSequenceFeature([], tf.string, default_value='',
                                                                                          allow_missing=True),
        'primary': tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0,
                                                                                          allow_missing=True),
        'tertiary': tf.io.FixedLenSequenceFeature([NUM_DIMENSIONS], tf.float32, default_value=0,
                                                                                          allow_missing=True),
        'evolutionary': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0,
                                                                                          allow_missing=True),
    }
    record = tf.io.parse_single_example(example_proto, feature_description)
    record['primary_onehot'] = tf.one_hot(record['primary'], NUM_AMINO_ACIDS)
    true_points = record['tertiary']
    mask = tf.cast(tf.reduce_all(true_points != 0.0, axis=-1), dtype=tf.float32)
    record['mask'] = mask
    mask2d = tf.tensordot(mask, mask, axes=0)
    distances = per_residue_distance(true_points * tf.expand_dims(mask, -1))
    distances *= mask2d

    record['true_distances'] = distances
    record['distance_mask'] = mask2d

    return record

def load_preprocessed_data(data_folder, filename):
    file_path = data_folder+filename
    if not os.path.exists(file_path):
        raise(ValueError(f"no data found in: {file_path}"))

    raw_dataset = tf.data.TFRecordDataset(file_path)
    decoded_dataset = raw_dataset.map(decode_preprocessed_fn)
    #debug info
    # data_size = sum(1 for _ in decoded_dataset)
    # print(f"Actual data size: {data_size}")
    for record in decoded_dataset.take(1):
        for key, value in record.items():
            print(key,"shape=",value.shape)
        print("Value info of one data record, structure:",record['tertiary'][0])

    return decoded_dataset

def display_two_structures(structure1, structure2, mask):
    dist1 = tf.squeeze(structure1) * mask
    dist2 = tf.squeeze(structure2) * mask

    f, axarr = plt.subplots(1, 2, figsize=(16, 8))
    axarr[0].set_title('predicted')
    axarr[1].set_title('true')
    im1 = axarr[0].imshow(dist1)
    im2 = axarr[1].imshow(dist2)

    f.colorbar(im1, ax=axarr[0])
    f.colorbar(im2, ax=axarr[1])
    plt.show()

def display_three_loss(avg_loss_list, avg_mse_loss_list, validate_loss_list):
    # 创建一个包含3个子图的绘图
    fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    # 绘制 Average Loss
    axs[0].plot(avg_loss_list, color='b', label='Average Loss')
    axs[0].set_ylabel('Average Loss')
    axs[0].legend()
    axs[0].grid(True)

    # 绘制 Average MSE Loss
    axs[1].plot(avg_mse_loss_list, color='r', label='Average MSE Loss')
    axs[1].set_ylabel('Average MSE Loss')
    axs[1].legend()
    axs[1].grid(True)

    # 绘制 Validate Loss
    axs[2].plot(validate_loss_list, color='g', label='Validate Loss')
    axs[2].set_xlabel('Batch')
    axs[2].set_ylabel('Validation Loss')
    axs[2].legend()
    axs[2].grid(True)

    # 设置总的标题
    fig.suptitle('Loss vs MSE Loss vs Validation Loss over Batch', fontsize=16)
    plt.show()


def display_test_loss_epochs(test_loss_epochs):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(test_loss_epochs) + 1), test_loss_epochs, marker='o', linestyle='-', color='b', label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Test Loss over Epochs')
    plt.grid(True)
    plt.legend()
    plt.show()

