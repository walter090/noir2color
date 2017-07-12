import os

import numpy as np
import scipy.misc
from PIL import Image
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn.preprocessing import MinMaxScaler


def load_image(image_file, output_size=(256, 256)):
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
        ratio and desired ratio differs greatly. Therefore, the solution here is to
        crop the image by the longer side and then resize it to the desired output size.

        Args:
            image(Image): Image object, the image to be cropped
            size(tuple): desired output size

        Returns:
            numpy array, re-sized image
        """
        # find the length of the short side
        desired_aspect_ratio = size[0] / size[1]
        aspect_ratio = image.size[1] / image.size[0]

        short_side_length = min(image.size)
        long_side_length = max(image.size)
        short_side = image.size.index(short_side_length)
        crop_size = [0, 0]
        if not np.sign(aspect_ratio - 1) == np.sign(desired_aspect_ratio - 1):
            crop_size[short_side] = short_side_length
            crop_size[1 - short_side] = short_side_length * min(size) / max(size)
        elif max(size) / min(size) > max(image.size) / min(image.size):
            crop_size[1 - short_side] = long_side_length
            crop_size[short_side] = long_side_length * min(size) / max(size)
        else:
            crop_size[short_side] = short_side_length
            crop_size[1 - short_side] = short_side_length * max(size) / min(size)

        cropped_img = image.crop((0, 0,)+tuple(crop_size))
        return cropped_img

    img = Image.open(image_file)
    img = crop(img)
    img.load()
    img_as_list = np.asarray(img, dtype='int32').astype('uint8')
    resized_img = resize(img_as_list, output_size, mode='wrap')
    return resized_img


def convert(folder, dest='img_np', bw_dest='img_bw', size=(256, 256)):
    """Convert jpg files to numpy array.
    Save images as jpeg to disk.

    Args:
        folder(str): Folder where images are stores.
        dest(str): Destination of converted csv files.
        bw_dest(str): Destination for black and white images.
        size(tuple): Size of the output.

    Returns:
        None
    """
    img_list = os.listdir(folder)
    if not os.path.isdir(dest):
        os.mkdir(dest)
    if not os.path.isdir(bw_dest):
        os.mkdir(bw_dest)

    # Auto incremental id for images
    img_id = 0
    for img in img_list:
        try:
            img_asarray = load_image(os.path.join(folder, img), size)
        except IOError:
            print('Cannot load image {}'.format(img))
            continue

        img_bw = color2bw(img_asarray)

        extension = '.jpg'
        scipy.misc.toimage(img_asarray).save(os.path.join(dest, str(img_id)) + extension)
        scipy.misc.toimage(img_bw).save(os.path.join(bw_dest, str(img_id)) + extension)

        img_id += 1


def scale(img, feature_range=(0, 255), target_range=(-1, 1)):
    """Pre-processing the RGB image
    Simple rescaling to the range [-1, 1]

    Args:
        img(numpy.ndarray): Natural image in numpy array.
        feature_range: Min max range of the numpy array.
        target_range: Min max range of the desired output.

    Returns:
        Scaled image
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.fit([[target_range[0]], [target_range[1]]])
    img = scaler.transform(img)

    return img.astype(np.float32)


def color2bw(img):
    """Convert a RGB image to a single channel black and white image
    Converted image is already scaled to [0, 1] range.

    Args:
        img(list): 3-D numpy array, RGB image

    Returns:
        Converted image with gray scale channel
    """
    return rgb2gray(img)
