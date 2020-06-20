import tensorflow_probability as tfp
import tensorflow as tf
import tree
from tensorflow_probability import distributions as tfpd


def cast_and_concat(x):
    """Concatenate every tensor in a nested list that contains
    many tensors with different shapes

    Args:

    x: list of tf.dtypes.float32
        list of tensors to be concatenated on the last axis

    Returns:

    y: tf.dtypes.float32
        a single tensor, the result of concatenating elements in x
    """

    x = tree.map_structure(
        lambda element: tf.cast(element, tf.float32), x)
    return tf.concat(tree.flatten(x), axis=-1)


class FeedForward(tf.keras.Sequential):

    def __init__(self, hidden_size, output_size):
        """Create a feed forward neural network with the provided
        hidden size and output size

        Args:

        hidden_size: tf.dtypes.int32
            the number of neurons in each hidden layer of the network
        output_size: tf.dtypes.int32
            the number of neurons in the output layer of the network
        """

        super(FeedForward, self).__init__([
            tf.keras.layers.Lambda(cast_and_concat),
            tf.keras.layers.Dense(hidden_size),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(hidden_size),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(output_size)])


class FeedForwardTanhGaussian(tf.keras.Sequential):

    def __init__(self, hidden_size, output_size, low, high):
        """Create a feed forward neural network with the provided
        hidden size and output size

        Args:

        hidden_size: tf.dtypes.int32
            the number of neurons in each hidden layer of the network
        output_size: tf.dtypes.int32
            the number of neurons in the output layer of the network
        """

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.low = low
        self.high = high

        super(FeedForwardTanhGaussian, self).__init__([
            tf.keras.layers.Lambda(cast_and_concat),
            tf.keras.layers.Dense(hidden_size),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(hidden_size),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(output_size * 2)])

    def get_distribution(self, x):
        """Build a tanh normalized gaussian distribution from the
        outputs of a neural network

        Args:

        x: tf.dtypes.float32
            the output of a neural network used to build a distribution

        Returns:

        d: tfp.Distribution
            a tanh normalized probability distribution over actions
        """

        loc = x[..., :self.output_size]
        scale_diag = tf.math.softplus(x[..., self.output_size:])
        tanh = tfp.bijectors.Tanh()
        tanh._is_constant_jacobian = True  # hack to get mean() to work

        chain = tfp.bijectors.Chain([
            tanh,
            tfp.bijectors.Scale((self.high - self.low) / 2.0),
            tfp.bijectors.Shift((self.high + self.low) / 2.0)])
        return tfpd.TransformedDistribution(
            tfpd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag),
            chain)
