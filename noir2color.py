import os
from random import shuffle

import numpy as np
import tensorflow as tf


def conv_avg_pool(x,
                  conv_ksize,
                  out_channels,
                  conv_stride,
                  pool_ksize=None,
                  pool_stride=None,
                  alpha=5,
                  name='conv',
                  padding='SAME'):
    """Convolution-LReLU-average pooling layers.

    This function takes the input and returns the output of the result after
    a convolution layer and an optional average pooling layer.

    Args:
        x(Tensor): Input from the previous layer.
        conv_ksize: 2-D tuple, filter size.
        out_channels: Out channels for the convnet.
        conv_stride: Stride for the convolution layer.
        pool_ksize: Filter size for the average pooling layer.
        pool_stride: Stride for the average pooling layer.
        alpha: Parameter for Leaky ReLU
        name: Name of the variable scope.
        padding: Padding for the layers, default 'SAME'.

    Returns:
        Output tensor
    """
    with tf.variable_scope(name):
        weights = tf.get_variable(name='conv_w',
                                  shape=[conv_ksize[0], conv_ksize[1], x.get_shape()[3], out_channels],
                                  initializer=tf.truncated_normal_initializer(stddev=0.02))
        bias = tf.get_variable(name='conv_b',
                               shape=[out_channels],
                               initializer=tf.zeros_initializer())

        conv_stride = (1,) + conv_stride + (1,)

        convoluted = tf.nn.conv2d(x, filter=weights, strides=conv_stride, padding=padding)
        convoluted = convoluted + bias
        conv = lrelu(convoluted, alpha)

        if pool_ksize is not None and pool_stride is not None:
            pool_ksize = (1,) + pool_ksize + (1,)
            pool_stride = (1,) + pool_stride + (1,)
            conv = tf.nn.avg_pool(conv, ksize=pool_ksize, strides=pool_stride, padding=padding)
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


def fully_conn(x, num_output, name='fc', activation=True):
    """Fully connected layer, this is is last parts of convnet.
    Fully connect layer requires each image in the batch be flattened.

    Args:
        x(Tensor): Input from the previous layer.
        num_output(int): Output size of the fully connected layer.
        name(str): Name for the fully connected layer variable scope.
        activation(bool): Set to True to add a leaky relu after fully connected
            layer. Set this argument to False if this is the final layer.

    Returns:
        Output tensor.
    """
    with tf.variable_scope(name):
        weights = tf.get_variable(name='fc_w', shape=[x.get_shape()[-1], num_output],
                                  initializer=tf.truncated_normal_initializer(stddev=0.02))
        biases = tf.get_variable(name='fc_b', shape=[num_output],
                                 initializer=tf.zeros_initializer())

        output = tf.nn.bias_add(tf.matmul(x, weights), biases)

        if activation:
            output = lrelu(output)

        return output


def deconv(x, ksize, out_channels, stride, output_shape=None, padding='SAME', name='deconv'):
    """Deconvolution (convolution transpose) layer.

    Args:
        x(Tensor): Input tensor from the previous layer.
        ksize: Filter size.
        out_channels: Filter number.
        stride: Stride size.
        output_shape: 1-D array, output size of the deconv layer. Default None,
            if this argument is left as None, an output shape will be calculated.
        padding: Padding method for the deconvolution, choose between 'SAME' and
            'VALID', default 'SAME' padding.
        name: Name for the variable scope of this layer.

    Returns:
        Output tensor.
    """
    with tf.variable_scope(name):
        weights = tf.get_variable(name='deconv_w',
                                  shape=[ksize[0], ksize[1], out_channels, x.get_shape()[3]],
                                  initializer=tf.truncated_normal_initializer(stddev=0.02))
        biases = tf.get_variable(name='deconv_b',
                                 shape=[out_channels],
                                 initializer=tf.zeros_initializer())

        stride = (1,) + stride + (1,)

        if output_shape is None:
            # if output_shape is not provided, compute default value
            stride_h = stride[1]
            stride_w = stride[2]
            input_h = x.get_shape()[1]
            input_w = x.get_shape()[2]
            filter_h = ksize[0]
            filter_w = ksize[1]

            output_shape = [n for n in x.get_shape()]
            output_shape[-1] = out_channels
            output_shape[-1] = ksize[-1]  # number of kernels

            if padding == 'SAME':
                output_shape[1] = input_h * stride_h
                output_shape[2] = input_w * stride_w
            elif padding == 'VALID':
                output_shape[1] = (input_h - 1) * stride_h + filter_h
                output_shape[2] = (input_w - 1) * stride_w + filter_w
            else:
                # if padding is not one of 'SAME' and 'VALID', raise an error
                raise ValueError("Padding must be one of 'SAME' and 'VALID', set to None to use"
                                 "default padding")

        deconvolved = tf.nn.conv2d_transpose(x,
                                             filter=weights, output_shape=output_shape,
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
    colored_images = tf.convert_to_tensor(img_list, dtype=tf.string)
    bw_images = tf.convert_to_tensor(bw_img_list, dtype=tf.string)

    # Partition images into training and testing
    partition = [0] * total_samples
    partition[: total_test_size] = [1] * total_test_size
    shuffle(partition)

    train_colored_images, test_colored_images = \
        tf.dynamic_partition(colored_images, partition, num_partitions=2)
    train_bw_images, test_bw_images = \
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

        # decode_jpeg somehow does not return shape of the image, need to manually set.
        # Make sure bw_img and colored_img are on the same rank as they may be concatenated
        # in the future.
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


def discriminator(input_x, base_x, reuse_variables=False, name='discriminator'):
    """Builds the discriminator part of the GAN.

    The discriminator takes two inputs, input_x, and base_x; input_x is the image
    for the network to judge whether it is fake or real (generated or original), while
    base_x is the conditional input, in this case base_x is the black-and-white image.

    Args:
        input_x: Candidate image to be judged by the discriminator.
        base_x: BW image the judgement is based on.
        reuse_variables: Set to True to reuse variables.
        name: Variable scope name.

    Returns:
        An unscaled value of the discriminator result.
    """
    with tf.variable_scope(name, reuse=reuse_variables):
        joint_x = tf.concat([input_x, base_x], axis=3)
        conv_1 = conv_avg_pool(joint_x,
                               conv_ksize=(4, 4),
                               out_channels=32,
                               conv_stride=(1, 1),
                               pool_ksize=(8, 8),
                               pool_stride=(2, 2))
        conv_2 = conv_avg_pool(conv_1,
                               conv_ksize=(4, 4),
                               out_channels=64,
                               conv_stride=(2, 2),
                               pool_ksize=(4, 4),
                               pool_stride=(2, 2))
        conv_3 = conv_avg_pool(conv_2,
                               conv_ksize=(4, 4),
                               out_channels=64,
                               conv_stride=(2, 2),
                               pool_ksize=(2, 2),
                               pool_stride=(1, 1))
        flat = flatten(conv_3)
        fc_1 = fully_conn(flat, num_output=1024)
        output = fully_conn(fc_1, num_output=1, activation=False)

        return output


def generator(input_x, name='generator', conv_layers=None, deconv_layers=None):
    """Generator network

    Args:
        input_x: Input image
        name: Variable scope name
        conv_layers: A list of lists specifying parameters for each conv layer.
        deconv_layers: A list of lists specifying parameters for each deconv layer.

    Returns:
        Generated image
    """
    with tf.variable_scope(name):
        if conv_layers is None:
            conv_layers = [
                # filter size, stride, output channels
                [(4, 4), (2, 2), 16],
                [(4, 4), (2, 2), 32],
                [(4, 4), (2, 2), 64],
                [(4, 4), (2, 2), 128],
                [(4, 4), (2, 2), 256],
            ]

        convolved = input_x
        for layer in conv_layers:
            convolved = conv_avg_pool(convolved, conv_ksize=layer[0],
                                      conv_stride=layer[1], out_channels=layer[2])
            convolved = batch_normalize(convolved)

        if deconv_layers is None:
            deconv_layers = [
                [],
            ]
