import tensorflow as tf
import numpy as np


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
        conv_ksize(list): ksize for the convolution layer. ksize should be in the
            shape of [filter_height, filter_width, in_channels, out_channels]
        conv_stride(list): stride for the convolution layer.
        pool_ksize(list): ksize for the average pooling layer.
        pool_stride(list): stride for the average pooling layer.
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
        ksize(list): filter size.
        stride(list): stride size.
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
                                             strides=stride, padding='SAME')
        deconv_out = tf.nn.bias_add(deconvolved, biases)
        return deconv_out


def batch_normalize(x):
    """Batch normalization for the network.

    Args:
        x: Input tensor from the previous layer.

    Returns:
        Output tensor.
    """
    # After conv layer, before activation
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
                                               variance_epsilon=1e-5)
        return normalized
