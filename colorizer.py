import tensorflow as tf
import noir2color
import warnings
import matplotlib.pyplot as plt
import numpy as np

from tools import image_process


def test_score(output, target):
    diff = np.abs(target - output)
    per_pixel_l1 = np.mean(diff)

    return per_pixel_l1


def model_test(meta, input_image=None, input_target=None,
               input_image_file=None, noise=True, z_dim=100):
    """Use the trained generator

    Function for using the trained generator, loads the trained weights and biases
    and use them in a new graph.

    Args:
        meta: string, checkpoint file
        input_image: 2-D or 3-D array, a single or list of input images.
        input_target: Optional target images for input, if provided, print
            the score.
        input_image_file: String of the name of the input image file,
            this argument is only used if input_image is None and is ignored when
            input_image is provided. This argument is present for when the user
            decides to run this function in the terminal.
        noise: Set True to add noise to the model. Set this argument according to
             the saved variables.
        z_dim: Depth of the noise. Set this argument according to the saved variables.

    Returns:
        gen_img: Colorized image(s)
    """
    session = tf.Session()

    # Use only input_image is both input_image and input_image_file are provided
    if input_image is not None:
        if input_image_file is not None:
            warnings.warn('input_image is not None, input_image_file ignored.',
                          UserWarning)
    elif input_image_file is not None:
        input_image = image_process.load_without_crop(input_image_file)
    else:
        raise ValueError('No image provided.')

    # Check if input_image contains multiple images
    shape = input_image.shape
    size = shape[0] if len(shape) == 3 else 1

    # Preprocess input image(s)
    base_img = tf.cast(tf.convert_to_tensor(input_image), dtype=tf.float32)
    base_img = noir2color.scale(base_img)
    base_img = tf.reshape(base_img, [size, shape[0], shape[1], 1])

    gen_tensor = noir2color.generator(input_x=base_img, testing=True,
                                      noise=noise, z_dim=z_dim)
    # Reuse restored variables
    tf.get_variable_scope().reuse_variables()

    saver = tf.train.Saver()
    saver.restore(session, meta)

    # Get generated image(s)
    gen_img = session.run(gen_tensor)

    # Scale the image to RGB format.
    gen_img = image_process.scale(gen_img,
                                  original_range=(-1, 1),
                                  target_range=(0, 255))

    # Show the image
    if input_image is None and input_image_file is not None:
        plt.axes('off')
        plt.imshow(gen_img)
        plt.show()

    if input_target is not None:
        print('Score: {}'.format(test_score(gen_img, input_target)))

    # Close the session
    session.close()

    return gen_img

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--meta', type=str, dest='meta',
                        help='Choose the file where the variables are stored.')
    parser.add_argument('-i', '--input', type=str, dest='input_image_file',
                        help='Specify the name of the black and white image file'
                             ' to feed the generator.')

    noise_parser = parser.add_mutually_exclusive_group()
    noise_parser.add_argument('--noise', action='store_true', dest='noise',
                              help='Provide noise to the generator')
    noise_parser.add_argument('--no-noise', action='store_true', dest='noise',
                              help='No input noise to the generator.')
    parser.set_defaults(noise=True)

    parser.add_argument('-z', '--z-dim', type=int, dest='z_dim',
                        help='Specify noise depth with an integer.')

    args = parser.parse_args()

    model_test(meta=args.meta, input_image_file=args.input_image_file,
               noise=args.noise, z_dim=args.z_dim)
