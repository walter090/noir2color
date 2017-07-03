import os

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from random import shuffle


def conv_avg_pool(x,
                  conv_ksize,
                  conv_stride,
                  pool_ksize,
                  pool_stride,
                  alpha=5,
                  pooling=True,
                  name='conv'):
    """Convolution-LReLU-average pooling layers.

    Args:
        x(Tensor): Input from the previous layer.
        conv_ksize(list): Filter size for the convolution layer. ksize should be in the
            shape of [filter_height, filter_width, in_channels, out_channels]
        conv_stride(list): Stride for the convolution layer.
        pool_ksize(list): Filter size for the average pooling layer.
        pool_stride(list): Stride for the average pooling layer.
        alpha(float): Parameter for Leaky ReLU
        pooling(bool): If set to False, a pooling layer will not be added after the conv
            layer and pooling parameters will be ignored
        name(str): Name of the variable scope.

    Returns:
        Output tensor
    """
    with tf.variable_scope(name):
        weights = tf.get_variable(name='conv_w', shape=conv_ksize,
                                  initializer=tf.truncated_normal_initializer(stddev=0.02))
        bias = tf.get_variable(name='conv_b', shape=[conv_ksize[-1]],
                               initializer=tf.zeros_initializer())

        convoluted = tf.nn.conv2d(x, filter=weights, strides=conv_stride, padding='VALID')
        convoluted = convoluted + bias
        conv = lrelu(convoluted, alpha)

        if pooling:
            conv = tf.nn.avg_pool(conv, ksize=pool_ksize, strides=pool_stride, padding='VALID')
        return conv


def lrelu(x, alpha=5):
    """Leaky ReLU activation.

    Args:
        x(Tensor): Input from the previous layer.
        alpha(float): Parameter for if x < 0.

    Returns:
        Output tensor
    """
    linear = 0.5 * x + 0.5 * tf.abs(x)
    leaky = 0.5 * alpha * x + 0.5 * alpha * tf.abs(x)
    output = leaky + linear
    return output


def flatten(x):
    """Flatten a tensor for the fully connected layer.
    Each image in a batch is flattened.

    Args:
        x(Tensor): 4-D tensor of shape [batch, height, width, channels] to be flattened
            to the shape of [batch, height * width * channels]

    Returns:
        Flattened tensor.
    """
    return tf.reshape(x, shape=[-1, np.prod(x[1:])])


def fully_conn(x, output_size, name='fc', activation=True):
    """Fully connected layer, this is is last parts of convnet.
    Fully connect layer requires each image in the batch be flattened.

    Args:
        x(Tensor): Input from the previous layer.
        output_size(int): Output size of the fully connected layer.
        name(str): Name for the fully connected layer variable scope.
        activation(bool): Set to True to add a leaky relu after fully connected
            layer.

    Returns:
        Output tensor.
    """
    with tf.variable_scope(name):
        weights = tf.get_variable(name='fc_w', shape=[x.get_shape()[-1], output_size],
                                  initializer=tf.truncated_normal_initializer(stddev=0.02))
        biases = tf.get_variable(name='fc_b', shape=[output_size],
                                 initializer=tf.zeros_initializer())

        output = tf.nn.bias_add(tf.matmul(x, weights), biases)

        if activation:
            output = lrelu(output)

        return output


def deconv(x, ksize, stride, output_shape=None, padding='SAME', name='deconv'):
    """Deconvolution (convolution transpose) layer.

    Args:
        x(Tensor): Input tensor from the previous layer.
        ksize(list): Filter size.
        stride(list): Stride size.
        output_shape(list): 1-D array, output size of the deconv layer. Default None,
            if this argument is left as None, an output shape will be calculated.
        padding(str): Padding method for the deconvolution, choose between 'SAME' and
            'VALID', default 'SAME' padding.
        name(str): Name for the variable scope of this layer.

    Returns:
        Output tensor.
    """
    with tf.variable_scope(name):
        weights = tf.get_variable(name='deconv_w', shape=ksize,
                                  initializer=tf.truncated_normal_initializer(stddev=0.02))
        biases = tf.get_variable(name='deconv_b', shape=ksize[-1],
                                 initializer=tf.zeros_initializer())

        if output_shape is None:
            # if output_shape is not provided, compute default value
            stride_h = stride[1]
            stride_w = stride[2]
            input_h = x.get_shape()[1]
            input_w = x.get_shape()[2]
            filter_h = ksize[0]
            filter_w = ksize[1]

            output_shape = [n for n in x.get_shape()]
            output_shape[-1] = ksize[-1]  # number of kernels

            if padding == 'SAME':
                output_shape[1] = (input_h + stride_h - 1) // stride_h
                output_shape[2] = (input_w + stride_w - 1) // stride_w
            elif padding == 'VALID':
                output_shape[1] = (input_h + stride_h - filter_h) // stride_h
                output_shape[2] = (input_w + stride_w - filter_w) // stride_w
            else:
                # if padding is not one of 'SAME' and 'VALID', raise an error
                raise ValueError("Padding must be one of 'SAME' and 'VALID', set to None to use"
                                 "default padding")

        deconvolved = tf.nn.conv2d_transpose(x, filter=weights, output_shape=output_shape,
                                             strides=stride, padding=padding)
        deconv_out = tf.nn.bias_add(deconvolved, biases)
        return deconv_out


def batch_normalize(x, epsilon=1e-5):
    """Batch normalization for the network.

    Args:
        x: Input tensor from the previous layer.
        epsilon: Variance epsilon.

    Returns:
        Output tensor.
    """
    # Before activation
    with tf.variable_scope('batch_norm'):
        mean, variance = tf.nn.moments(x, axes=[0])

        scale = tf.get_variable('bn_scale', shape=[x.get_shape()[-1]],
                                initializer=tf.random_normal_initializer())
        offset = tf.get_variable('bn_bias', shape=[x.get_shape()[-1]],
                                 initializer=tf.zeros_initializer())
        normalized = tf.nn.batch_normalization(x=x,
                                               mean=mean,
                                               variance=variance,
                                               offset=offset,
                                               scale=scale,
                                               variance_epsilon=epsilon)
        return normalized


def process_data(folder, bw_folder, test_size=0.1):
    """Read and partition data.
    This function should be run before the input pipeline.

    Args:
        folder: Directory to the unprocessed images.
        bw_folder: Directory to the black and white images.
        test_size: Test set size, float between 0 and 1, default 0.1

    Returns:
        A dictionary of tensors containing image file names.
    """
    def file_sort(file_name):
        return int(file_name.split('.')[0])

    img_list = sorted(os.listdir(folder), key=file_sort)
    bw_img_list = sorted(os.listdir(bw_folder), key=file_sort)
    img_list = [os.path.join(folder, img) for img in img_list]
    bw_img_list = [os.path.join(bw_folder, img) for img in bw_img_list]

    total_samples = len(img_list)
    total_test_size = int(test_size * total_samples)

    # List of image file names.
    colored_images = ops.convert_to_tensor(img_list, dtype=dtypes.string)
    bw_images = ops.convert_to_tensor(bw_img_list, dtype=dtypes.string)

    # Partition images into training and testing
    partition = [0] * total_samples
    partition[: total_test_size] = [1] * total_test_size
    shuffle(partition)

    train_colored_images, test_colored_images =\
        tf.dynamic_partition(colored_images, partition, num_partitions=2)
    train_bw_images, test_bw_images =\
        tf.dynamic_partition(bw_images, partition, num_partitions=2)

    return {'train': (train_bw_images, train_colored_images),
            'test': (test_bw_images, test_colored_images)}


def input_pipeline(images_tuple, height=256, width=256, batch_size=50):
    """Pipeline for inputting images.

    Args:
        images_tuple: Python dictionary containing string typed tensors that
            are image file names. The dictionary comes in the shape of
            {'train': (train_bw_images, train_colored_images),
            'test': (test_bw_images, test_colored_images)}
        height: Height of the image.
        width: Width of the image.
        batch_size: Size of each batch, default 50.

    Returns:
        A tuple of black and white image batch and colored image patch.
    """
    def read_image(input_queue_):
        """Read images from specified files.

        Args:
            input_queue_: Tensor of type string that contains image file names.

        Returns:
            Two tensors, black-and-white and colored images read from the files.
        """
        bw_img_file = tf.read_file(input_queue_[0])
        colored_img_file = tf.read_file(input_queue_[1])
        bw_img_ = tf.image.decode_jpeg(bw_img_file, channels=1)  # Decode as grayscale
        colored_img_ = tf.image.decode_jpeg(colored_img_file, channels=3)  # Decode as RGB

        # decode_jpeg somehow does not return shape of the image, need to manually set
        bw_img_.set_shape([height, width, 1])
        colored_img_.set_shape([height, width, 3])

        return bw_img_, colored_img_

    bw_images, colored_images = images_tuple

    # Create an input queue, a queue of string tensors that are image file names.
    input_queue = tf.train.slice_input_producer([bw_images, colored_images])

    bw_img, colored_img = read_image(input_queue)
    bw_batch, colored_batch = tf.train.batch([bw_img, colored_img],
                                             batch_size=batch_size)

    return bw_batch, colored_batch


def discriminator(input_x, base_x, reuse_variables=False):
    """Builds the discriminator part of the GAN.

    The discriminator takes two inputs, input_x, and base_x; input_x is the image
    for the network to judge whether it is fake or real (generated or original), while
    base_x is the condition, in this case base_x is the black-and-white image.

    Args:
        input_x: Candidate image to be judged by the discriminator.
        base_x: BW image the judgement is based on.
        reuse_variables: Set to True to reuse variables.

    Returns:
        An unscaled value of the discriminator result.
    """
    with tf.variable_scope('discriminator', reuse=reuse_variables):
        raise NotImplementedError
