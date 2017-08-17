import warnings

import numpy as np
import tensorflow as tf

import noir2color
from tools import image_process


def test_score(output, target, score='l1'):
    """Computes the test score of the generated images.

    Args:
        output: An array presentation of generated images.
        target: An array presentation of target images.
        score: Scoring method, defaults l2

    Returns:
        Test score
    """
    diff = np.abs(target - output)
    if score == 'l1':
        per_pixel_score = np.mean(diff)
    elif score == 'l2':
        per_pixel_score = np.mean([entry ** 2 for entry in diff])
    else:
        raise ValueError('No such option, choose between l1 and l2')

    return per_pixel_score


def model_test(meta,
               skip_conn=True,
               input_image=None,
               input_target=None,
               input_image_file=None,
               noise=True,
               z_dim=100):
    """Use the trained generator

    Function for using the trained generator, loads the trained weights and biases
    and use them in a new graph.

    Args:
        meta: string, checkpoint file
        skip_conn: Set True to use skip connection in generator.
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
        gen_images: Colorized image(s)
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
    input_image = np.array(input_image)
    shape = input_image.shape

    if len(shape) == 3:
        size = shape[0]
        shape = (shape[1], shape[2])
    else:
        size = 1

    # Preprocess input image(s)
    base_img = tf.cast(tf.convert_to_tensor(input_image), dtype=tf.float32)
    base_img = noir2color.scale(base_img)
    base_img = tf.reshape(base_img, [size, shape[0], shape[1], 1])

    gen_tensor = noir2color.generator(input_x=base_img,
                                      testing=True,
                                      noise=noise,
                                      z_dim=z_dim,
                                      skip_conn=skip_conn)

    # Reuse restored variables
    tf.get_variable_scope().reuse_variables()

    if meta is not None:
        saver = tf.train.Saver()
        saver.restore(session, meta)
    else:
        session.run(tf.global_variables_initializer())

    # Get generated image(s)
    gen_images = session.run(gen_tensor)

    # Scale the image to RGB format.
    gen_images = image_process.scale(gen_images,
                                     original_range=(-1, 1),
                                     target_range=(0, 255))

    score = None
    if input_target is not None:
        score = test_score(gen_images, input_target)
        print('Score: {}'.format(score))

    # Close the session
    session.close()

    return gen_images, score


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
