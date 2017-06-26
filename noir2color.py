import tensorflow as tf
import numpy as np


def conv_pool(x, conv_ksize, conv_stride, pool_ksize,
              pool_stride, alpha=5, pooling=True, name='conv'):
    """Convolution-LReLU-average pooling layers

    Args:
        x(Tensor): Input from the previous layer.
        conv_ksize(list): 4-D array, ksize for the convolution layer.
        conv_stride(list): 4-D array, stride for the convolution layer.
        pool_ksize(list): 4-D array, ksize for the average pooling layer.
        pool_stride(list): 4-D array, stride for the average pooling layer.
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
                               initializer=tf.constant_initializer(0))

        convoluted = tf.nn.conv2d(x, filter=weights, strides=conv_stride, padding='SAME')
        convoluted = convoluted + bias
        conv = lrelu(convoluted, alpha)

        if pooling:
            conv = tf.nn.avg_pool(conv, ksize=pool_ksize, strides=pool_stride, padding='SAME')
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
    return leaky + linear


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
                                 initializer=tf.constant_initializer(0))

        output = tf.nn.bias_add(tf.matmul(x, weights), biases)

        if activation:
            output = lrelu(output)

        return output


def deconv(x, ksize, stride, output_size, name='deconv'):
    """Deconvolution (convolution transpose) layer.

    Args:
        x(Tensor): Input tensor from the previous layer.
        ksize(list): 4-D array, filter size.
        stride(list): 4-D array, stride size.
        output_size(list): 1-D array, output size of the deconv layer.
        name(str): Name for the variable scope of this layer.

    Returns:
        Output tensor.
    """
    raise NotImplementedError
