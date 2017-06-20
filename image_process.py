from PIL import Image
from skimage.color import rgb2gray
import numpy as np


def load_image(image_file):
    """Function for loading images.
    This function uses Pillow to load image from file and the image
    is then converted to a numpy array for further processing

    Args:
        image_file(str): path of the image file

    Returns:
        img_as_list(list): a numpy array
    """
    img = Image.open(image_file)
    img.load()
    img_as_list = np.asarray(img, dtype='int32')
    return img_as_list


def rescale(img):
    """Pre-processing the RGB image
    Simple rescaling to the range [0, 1]

    Args:
        img(list): Natural image in numpy array

    Returns:
        Scaled image
    """
    return img / 255


def color2bw(img):
    """Convert a RGB image to a single channel black and white image

    Args:
        img(list): 3-D numpy array, RGB image

    Returns:
        Converted image with gray scale channel
    """
    return rgb2gray(img)
