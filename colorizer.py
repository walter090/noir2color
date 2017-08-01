import tensorflow as tf
import noir2color

from tools import image_process


def model_test(meta, input_image, noise=True, z_dim=100):
    """Use the trained generator

    Function for using the trained generator, loads the trained weights and biases
    and use them in a new graph.

    Args:
        meta: string, checkpoint file
        input_image: 2-D or 3-D array, a single or list of input images
        noise: Set True to add noise to the model
        z_dim: Depth of the noise.

    Returns:
        gen_img: Colorized image(s)
    """
    with tf.Session() as session:
        shape = input_image.shape
        size = shape[0] if len(shape) == 3 else 1

        base_img = tf.cast(tf.convert_to_tensor(input_image), dtype=tf.float32)
        base_img = noir2color.scale(base_img)
        base_img = tf.reshape(base_img, [size, shape[0], shape[1], 1])

        gen_tensor = noir2color.generator(input_x=base_img, testing=True,
                                          noise=noise, z_dim=z_dim)
        tf.get_variable_scope().reuse_variables()

        saver = tf.train.Saver()
        saver.restore(session, meta)

        gen_img = session.run(gen_tensor)

    # Scale the image to RGB format.
    gen_img = image_process.scale(gen_img,
                                  original_range=(-1, 1),
                                  target_range=(0, 255))

    return gen_img

# TODO Add command line argument parser
