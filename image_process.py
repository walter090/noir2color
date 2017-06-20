from PIL import Image
from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np


def load_image(image_file, output_size=(300, 400)):
    """Function for loading images.
    This function uses Pillow to load image from file and the image
    is then converted to a numpy array for further processing

    Args:
        image_file(str): path of the image file
        output_size(tuple): desired size of the image

    Returns:
        img_as_list(list): a numpy array
    """
    def crop(image, size=output_size):
        """Crop an image.
        To make sure all input images have the same size, we need to resize them.
        Simply use skimage's resize function may distort the image if the original aspect
        ration and desired ratio differs greatly. Therefore, the solution here is to
        crop the image by the longer side and then resize it to the desired output size.

        Args:
            image(Image): Image object, the image to be cropped
            size(tuple): desired output size

        Returns:

        """
        # calculate aspect ratio of desired output
        aspect_ratio = size[0] / size[1]
        # find the length of the short side
        short_side_length = min(image.size)
        short_side = image.size.index(short_side_length)
        crop_size = [0, 0]
        crop_size[short_side] = short_side_length
        crop_size[1 - short_side] = short_side_length * aspect_ratio if aspect_ratio > 1\
            else short_side_length / aspect_ratio
        cropped_img = image.crop((0, 0,)+tuple(crop_size))
        return cropped_img

    img = Image.open(image_file)
    img = crop(img)
    img.load()
    img_as_list = np.asarray(img, dtype='int32').astype('uint8')
    resized_img = resize(img_as_list, output_size)
    return resized_img


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
    Converted image is already scaled to [0, 1] range.

    Args:
        img(list): 3-D numpy array, RGB image

    Returns:
        Converted image with gray scale channel
    """
    return rgb2gray(img)
