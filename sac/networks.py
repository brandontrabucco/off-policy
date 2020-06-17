import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfpd
import tensorflow as tf
import tree


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

        tanh = tfp.bijectors.Tanh()
        tanh._is_constant_jacobian = True  # hack to get mean() to work
        self._chain = tfp.bijectors.Chain([
            tanh,
            tfp.bijectors.Scale((high - low) / 2.0),
            tfp.bijectors.Shift((high + low) / 2.0)])

        def make_distribution_fn(t):
            loc = t[..., :output_size]
            scale_diag = tf.math.softplus(t[..., output_size:])
            return tfpd.TransformedDistribution(
                tfpd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag),
                self._chain)

        super(FeedForwardTanhGaussian, self).__init__([
            tf.keras.layers.Lambda(cast_and_concat),
            tf.keras.layers.Dense(hidden_size),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(hidden_size),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(output_size * 2),
            tfp.layers.DistributionLambda(make_distribution_fn)])


def split_shift_log_scale(x):
    """Splits the input tensor into two tensors of equal length
    along the channels axis

    Args:

    x: tf.dtypes.float32
        a tensor that has an even number of channels in the last axis

    Returns:

    shift: tf.dtypes.float32
        the first half of the channels in the input tensor x
    log_scale: tf.dtypes.float32
        the first half of the channels in the input tensor x
    """

    return tf.split(x, 2, axis=-1)


class LatentSpacePolicy(tf.keras.Sequential):

    def __init__(self, hidden_size, output_size, low, high):
        """Creates a latent space policy by stacking many layers
        of the invertible RealNVP bijector

        Args:

        hidden_size: tf.dtypes.int32
            the number of neurons in each hidden layer of the network
        output_size: tf.dtypes.int32
            the number of neurons in the output layer of the network
        low: tf.dtypes.float32
            the minimum value of the output space of the network
        high: tf.dtypes.float32
            the maximum value of the output space of the network
        """

        self._chain = tfp.bijectors.Chain([
            tfp.bijectors.Tanh(),
            tfp.bijectors.Scale((high - low) / 2.0),
            tfp.bijectors.Shift((high + low) / 2.0)])

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.add_level()

        def make_distribution_fn(t):
            loc = tf.zeros([tf.shape(t)[0], output_size])
            return tfpd.TransformedDistribution(
                tfpd.MultivariateNormalDiag(loc=loc), self._chain)

        super(LatentSpacePolicy, self).__init__([
            tf.keras.layers.Lambda(cast_and_concat),
            tfp.layers.DistributionLambda(make_distribution_fn)])

    def make_template(self):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_size),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.hidden_size),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.output_size * 2),
            tf.keras.layers.Lambda(split_shift_log_scale)])

    def add_level(self):
        self._chain.bijectors.insert(0, tfp.bijectors.RealNVP(
            fraction_masked=+0.5,
            shift_and_log_scale_fn=self.make_template()))
        self._chain.bijectors.insert(0, tfp.bijectors.RealNVP(
            fraction_masked=-0.5,
            shift_and_log_scale_fn=self.make_template()))
