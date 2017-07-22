import tensorflow as tf
import noir2color


def model_test(meta, input_image):
    """Use the trained generator

    Function for using the trained generator, loads the trained weights and biases
    and use them in a new graph.

    Args:
        meta: string, checkpoint file
        input_image: 2-D or 3-D array, a single or list of input images

    Returns:
        gen_img: Colorized image(s)
    """
    with tf.Session() as session:
        shape = input_image.shape
        size = shape[0] if len(shape) == 3 else 1

        base_img = tf.cast(tf.convert_to_tensor(input_image), dtype=tf.float32)
        base_img = noir2color.scale(base_img)
        base_img = tf.reshape(base_img, [size, *shape, 1])

        gen_tensor = noir2color.generator(input_x=base_img)
        tf.get_variable_scope().reuse_variables()

        saver = tf.train.Saver()
        saver.restore(session, meta)

        gen_img = session.run(gen_tensor)

    return gen_img

# TODO Add command line argument parser
