from offpolicy.networks.feed_forward import FeedForward
from tensorflow_probability import distributions as tfpd
import tensorflow as tf


class Gaussian(FeedForward):

    def __init__(self, low, high, input_size, hidden_size, output_size,
                 exploration_noise=0.1):
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
        self.exploration_noise = exploration_noise

        super(FeedForward, self).__init__([
            tf.keras.layers.Dense(hidden_size, input_shape=(input_size,)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(hidden_size),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(output_size),
            tf.keras.layers.Activation("tanh")])

    def get_distribution(self, inputs, **kwargs):
        """Build a normalized gaussian distribution using a neural
        network backbone and return the distribution

        Args:

        inputs: tf.dtypes.float32
            the output of a neural network used to build a distribution

        Returns:

        d: tfp.Distribution
            a tensorflow probability distribution over normalized actions
        """

        loc = self.__call__(inputs, **kwargs)
        loc = loc * (self.high - self.low) / 2.0
        loc = loc + (self.high + self.low) / 2.0
        scale_diag = tf.ones_like(loc) * self.exploration_noise
        return tfpd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

    def mean(self, inputs, log_probs=False, **kwargs):
        """Build a normalized gaussian distribution using a neural
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
        samples = tf.clip_by_value(d.mean(), self.low, self.high)
        return (samples, d.log_prob(
            samples)) if log_probs else samples

    def sample(self, inputs, log_probs=False, **kwargs):
        """Build a normalized gaussian distribution using a neural
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
        samples = tf.clip_by_value(d.sample(), self.low, self.high)
        return (samples, d.log_prob(
            samples)) if log_probs else samples
