import tensorflow as tf
import tree


def concat(x):
    """Concatenate every tensor in a nested list that contains
    many tensors with different shapes

    Args:

    x: list of tf.dtypes.float32
        list of tensors to be concatenated on the last axis

    Returns:

    y: tf.dtypes.float32
        a single tensor, the result of concatenating elements in x
    """

    return tf.concat(tree.flatten(tree.map_structure(
        lambda element: tf.cast(element, tf.float32), x)), axis=-1)


class FeedForward(tf.keras.Sequential):

    def __call__(self, inputs, **kwargs):
        return super(FeedForward, self).__call__(concat(inputs), **kwargs)

    def __init__(self, input_size, hidden_size, output_size):
        """Create a feed forward neural network with the provided
        hidden size and output size

        Args:

        hidden_size: tf.dtypes.int32
            the number of neurons in each hidden layer of the network
        output_size: tf.dtypes.int32
            the number of neurons in the output layer of the network
        """

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        hidden_init = tf.keras.initializers.VarianceScaling(
            scale=1.0 / 3.0, mode='fan_in', distribution='uniform')
        out_init = tf.random_uniform_initializer(
            minval=-0.003, maxval=0.003, seed=None)

        super(FeedForward, self).__init__([
            tf.keras.layers.Dense(hidden_size,
                                  input_shape=(input_size,),
                                  kernel_initializer=hidden_init),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(hidden_size,
                                  kernel_initializer=hidden_init),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(output_size,
                                  kernel_initializer=out_init)])
