import tensorflow as tf
import numpy as np


def conv_pool(input, conv_ksize, conv_stride,
              out_channels, pool_ksize, pool_stride,
              alpha=5, pooling=True, name='conv'):
    """Convolution-LReLU-average pooling layers

    Args:
        input(Tensor): Input from the previous layer.
        conv_ksize(list): 4-D array, ksize for the convolution layer.
        conv_stride(list): 4-D array, stride for the convolution layer.
        out_channels(int): Number of output channels for the convolution layers, i.e.
            number of filters.
        pool_ksize(list): 4-D array, ksize for the average pooling layer.
        pool_stride(list): 4-D array, stride for the average pooling layer.
        alpha(float): Parameter for Leaky ReLU
        pooling(bool): If set to False, a pooling layer will not be added after the conv
            layer and pooling parameters will be ignored
        name(str): Name of the variable scope.

    Returns:
        Output tensor
    """
    raise NotImplementedError


def lrelu(input, alpha, name='lrelu'):
    """Leaky ReLU activation.

    Args:
        input(Tensor): Input from the previous layer.
        alpha(int): Parameter for if x < 0.
        name(str): Name for the variable scope.

    Returns:
        Output tensor
    """
    raise NotImplementedError


def fully_conn(input, name='fully_conn'):
    """Fully connected layer, this is is last parts of convnet.
    Fully connect layer requires each image in the batch be flattened.

    Args:
        input(Tensor): Input from the previous layer.
        name(str): Name for the fully connected layer variable scope.

    Returns:
        Output tensor.
    """
    raise NotImplementedError


def deconv(input):
    raise NotImplementedError
