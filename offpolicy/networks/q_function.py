import tensorflow as tf
import tree


def cast(x):
    """Concatenate every tensor in a nested list that contains
    many tensors with different shapes

    Args:

    x: list of tf.dtypes.float32
        list of tensors to be concatenated on the last axis

    Returns:

    y: tf.dtypes.float32
        a single tensor, the result of concatenating elements in x
    """

    return tree.map_structure(
        lambda element: tf.cast(element, tf.float32), x)


class QFunction(tf.keras.Model):

    def __call__(self, inputs, **kwargs):
        return super(QFunction, self).__call__(cast(inputs), **kwargs)

    def __init__(self, obs_size, act_size):
        """Create a feed forward neural network with the provided
        hidden size and output size

        Args:

        hidden_size: tf.dtypes.int32
            the number of neurons in each hidden layer of the network
        output_size: tf.dtypes.int32
            the number of neurons in the output layer of the network
        """

        hidden_init = tf.keras.initializers.VarianceScaling(
            scale=1.0 / 3.0, mode='fan_in', distribution='uniform')
        out_init = tf.random_uniform_initializer(
            minval=-0.003, maxval=0.003, seed=None)

        obs = tf.keras.Input(shape=(obs_size,))
        act = tf.keras.Input(shape=(act_size,))

        h = tf.keras.layers.Concatenate(axis=-1)([obs, act])
        h = tf.keras.layers.Dense(
            400, kernel_initializer=hidden_init)(h)
        h = tf.keras.layers.Dense(
            300, kernel_initializer=hidden_init)(h)
        h = tf.keras.layers.Dense(
            1, kernel_initializer=out_init)(h)

        super(QFunction, self).__init__(inputs=[obs, act], outputs=h)


class QFunction2(tf.keras.Model):

    def __call__(self, inputs, **kwargs):
        return super(QFunction2, self).__call__(cast(inputs), **kwargs)

    def __init__(self, obs_size, act_size):
        """Create a feed forward neural network with the provided
        hidden size and output size

        Args:

        hidden_size: tf.dtypes.int32
            the number of neurons in each hidden layer of the network
        output_size: tf.dtypes.int32
            the number of neurons in the output layer of the network
        """

        hidden_init = tf.keras.initializers.VarianceScaling(
            scale=1.0 / 3.0, mode='fan_in', distribution='uniform')
        out_init = tf.random_uniform_initializer(
            minval=-0.003, maxval=0.003, seed=None)

        obs = tf.keras.Input(shape=(obs_size,))
        act = tf.keras.Input(shape=(act_size,))

        h = tf.keras.layers.Concatenate(axis=-1)([obs, act])
        h = tf.keras.layers.Dense(
            400, kernel_initializer=hidden_init)(h)

        h = tf.keras.layers.Concatenate(axis=-1)([act, h])
        h = tf.keras.layers.Dense(
            300, kernel_initializer=hidden_init)(h)
        h = tf.keras.layers.Dense(
            1, kernel_initializer=out_init)(h)

        super(QFunction2, self).__init__(inputs=[obs, act], outputs=h)
