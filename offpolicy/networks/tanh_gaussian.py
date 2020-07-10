from offpolicy.networks.feed_forward import FeedForward
from tensorflow_probability import distributions as tfpd
import tensorflow_probability as tfp
import tensorflow as tf


def split(x):
    """split a tensor in half to parameterize the mean and standard
    deviation of a multivariate gaussian

    Args:

    x: tf.dtypes.float32
        the activations from the final layer of a neural network

    Returns:

    mean: tf.dtypes.float32
        the mean of a probability distribution
    std: tf.dtypes.float32
        the standard deviation of a probability distribution
    """

    mean, std = tf.split(x, num_or_size_splits=2, axis=-1)
    return mean, tf.math.softplus(std)


class TanhGaussian(FeedForward):

    def __init__(self, low, high, input_size, hidden_size, output_size):
        """Create a feed forward neural network with the provided
        hidden size and output size

        Args:

        hidden_size: tf.dtypes.int32
            the number of neurons in each hidden layer of the network
        output_size: tf.dtypes.int32
            the number of neurons in the output layer of the network
        """

        self.low = low
        self.high = high
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.scale = (self.high - self.low)[tf.newaxis] / 2.0
        self.shift = (self.high + self.low)[tf.newaxis] / 2.0

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
            tf.keras.layers.Dense(output_size * 2,
                                  kernel_initializer=out_init)])

    def get_distribution(self, inputs, **kwargs):
        """Build a tanh normalized gaussian distribution using a neural
        network backbone and return the distribution

        Args:

        inputs: tf.dtypes.float32
            the output of a neural network used to build a distribution

        Returns:

        d: tfp.Distribution
            a tensorflow probability distribution over normalized actions
        """

        loc, scale_diag = split(self.__call__(inputs, **kwargs))
        d = tfpd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)
        tanh_bijector = tfp.bijectors.Tanh()
        tanh_bijector._is_constant_jacobian = True
        return tfpd.TransformedDistribution(d, tfp.bijectors.Chain([
            tfp.bijectors.Shift(shift=self.shift),
            tfp.bijectors.Scale(scale=self.scale),
            tanh_bijector]))

    def mean(self, inputs, log_probs=False, **kwargs):
        """Build a tanh normalized gaussian distribution using a neural
        network backbone and return the mean

        Args:

        inputs: tf.dtypes.float32
            the output of a neural network used to build a distribution

        Returns:

        mean: tf.dtypes.float32
            the mean of a probability distribution over actions
        log_pis: tf.dtypes.float32
            (optional) the log probabilities of the sampled actions
        """

        d = self.get_distribution(inputs, **kwargs)
        samples = tf.clip_by_value(
            d.mean(), self.low[tf.newaxis], self.high[tf.newaxis])
        return (samples, d.log_prob(
            samples)) if log_probs else samples

    def sample(self, inputs, log_probs=False, **kwargs):
        """Build a tanh normalized gaussian distribution using a neural
        network backbone and return samples

        Args:

        inputs: tf.dtypes.float32
            the output of a neural network used to build a distribution

        Returns:

        samples: tf.dtypes.float32
            samples of a probability distribution over actions
        log_pis: tf.dtypes.float32
            (optional) the log probabilities of the sampled actions
        """

        d = self.get_distribution(inputs, **kwargs)
        samples = tf.clip_by_value(
            d.sample(), self.low[tf.newaxis], self.high[tf.newaxis])
        return (samples, d.log_prob(
            samples)) if log_probs else samples
