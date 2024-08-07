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
	num_examples = None
	in_height = None
	in_width = None
	input_in_channels = None

	filter_height = None
	filter_width = None
	filter_in_channels = None
	filter_out_channels = None

	num_examples_stride = None
	strideY = None
	strideX = None
	channels_stride = None

	# Cleaning padding input

	# Calculate output dimensions

	pass


  