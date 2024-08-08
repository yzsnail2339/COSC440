from __future__ import absolute_import

import os
import tensorflow as tf
import numpy as np
import random
import math

def conv2d(inputs, filters, strides, padding):
	"""
	Performs 2D convolution given 4D inputs and filter Tensors.
	:param inputs: tensor with shape [num_examples, in_height, in_width, in_channels]
	:param filters: tensor with shape [filter_height, filter_width, in_channels, out_channels]
	:param strides: MUST BE [1, 1, 1, 1] - list of strides, with each stride corresponding to each dimension in input
	:param padding: either "SAME" or "VALID", capitalization matters
	:return: outputs, NumPy array or Tensor with shape [num_examples, output_height, output_width, output_channels]
	"""
	num_examples = inputs.shape[0]
	in_height = inputs.shape[1]
	in_width = inputs.shape[2]
	input_in_channels = inputs.shape[3]

	filter_height = filters.shape[0]
	filter_width = filters.shape[1]
	filter_in_channels = filters.shape[2]
	filter_out_channels = filters.shape[3]

	num_examples_stride = strides[0]
	strideY = strides[1]
	strideX = strides[2]
	channels_stride = strides[3]

	# Cleaning padding input
	assert  input_in_channels == filter_in_channels, f"number of channels in input filters is {input_in_channels} and inputs is {filter_in_channels}, not equivalent"
	pad_h, pad_w = 0, 0
	if padding == 'SAME':
		pad_h = int((filter_height - 1)//2)
		pad_w = int((filter_width - 1)//2)
		pad_width = ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0))
		inputs = np.pad(inputs, pad_width=pad_width, mode="constant",constant_values=0)
	
	# Calculate output dimensions
	output_h = int((in_height + 2*pad_h - filter_height) // strideY + 1)
	output_w = int((in_width + 2*pad_w - filter_width) // strideX + 1)
	output_result = np.zeros((num_examples, output_h, output_w, filter_out_channels))
	
	# for n in range(num_examples):
	# 	for c_in in range(input_in_channels):
	# 		for h in range(output_h):
	# 			for w in range(output_w):
	# 				for c_out in range(filter_out_channels):
	# 					output_result[n,h,w,c_out] += np.sum(inputs[n, h: h + filter_height, w: w + filter_width, c_in] * filters[:, :, c_in, c_out])
	# return output_result

	for n in range(num_examples):
		for h in range(output_h):
			for w in range(output_w):
				for c_out in range(filter_out_channels):
					output_result[n,h,w,c_out] += np.sum(inputs[n, h: h + filter_height, w: w + filter_width, :] * filters[:, :, : , c_out])
	return output_result

if __name__ == "__main__":
    imgs = np.array([[2, 2, 3, 3, 3], [0, 1, 3, 0, 3], [2, 3, 0, 1, 3], [3, 3, 2, 1, 2], [3, 3, 0, 2, 3], [2, 2, 3, 3, 3], [0, 1, 3, 0, 3], [2, 3, 0, 1, 3], [3, 3, 2, 1, 2], [3, 3, 0, 2, 3]],
                    dtype=np.float32)
    imgs = np.reshape(imgs, (1, 5, 5, 2))
    filters = tf.Variable(tf.random.truncated_normal([2, 2, 2, 2],
                                                     dtype=tf.float32,
                                                     stddev=1e-1),
                          name="filters")
    my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="SAME")
    print(my_conv)
    tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="SAME")
    print(tf_conv)
    print(my_conv.shape)
    print(tf_conv.shape)
    print(my_conv == tf_conv)