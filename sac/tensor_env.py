import tensorflow as tf
import tensorflow_probability as tfp
import gym


class Box(gym.spaces.Space):

    def __init__(self, low, high):
        """Create a space that supports tensorflow operations
        and sampling tensorflow tensors

        Args:

        low: tf.dtypes.float32
            a tensor that specifies the lower bound for the space
        high: tf.dtypes.float32
            a tensor that specifies the upper bound for the space
        """

        super(Box, self).__init__(None, None)
        self.low = tf.Variable(low, dtype=tf.float32)
        self.high = tf.Variable(high, dtype=tf.float32)
        self.shape = self.low.get_shape()
        self.dtype = self.low.dtype

        # check if the space is upper and lower bounded
        self.lb = tf.logical_not(tf.math.is_nan(self.low))
        self.ub = tf.logical_not(tf.math.is_nan(self.high))

        # create sampling distributions for the space
        self.exponential = tfp.distributions.Exponential(tf.ones_like(self.low))
        self.uniform = tfp.distributions.Uniform(self.low, self.high)
        self.normal = tfp.distributions.Normal(
            tf.zeros_like(self.low), tf.ones_like(self.low))

    @tf.function
    def sample(self):
        """Draw samples from a probability distribution whose support is
        the entire space defined by low and high
        """

        samples = self.normal.sample()
        samples = tf.where(
            self.lb, self.exponential.sample() + self.low, samples)
        samples = tf.where(
            self.ub, -self.exponential.sample() + self.high, samples)
        return tf.where(
            tf.logical_and(self.lb, self.ub), self.uniform.sample(), samples)

    @tf.function
    def contains(self, x):
        """Check if a tensorflow tensor is contained within the upper
        and lower bounds of this space

        Args:

        x: tf.dtypes.float32
            a tensor that shall be compared with the upper and lower bounds
        """

        return tf.logical_and(
            tf.greater_equal(x, self.low), tf.less_equal(x, self.high))


class TensorEnv(gym.Env):

    def __init__(self, wrapped_env):
        """Create an in-graph environment with the same API as the gym.Env
        class, but with static graph operations

        Args:

        wrapped_env: gym.Env
            an OpenAI Gym environment with step, reset, and render functions
        """

        self.wrapped_env = wrapped_env
        self.observation_space = Box(self.wrapped_env.observation_space.low,
                                     self.wrapped_env.observation_space.high)
        self.context_space = Box(self.wrapped_env.context_space.low,
                                 self.wrapped_env.context_space.high)
        self.action_space = Box(self.wrapped_env.action_space.low,
                                self.wrapped_env.action_space.high)
        self.reward_range = tf.convert_to_tensor(
            self.wrapped_env.reward_range)

    def _step(self, action):
        return self.wrapped_env.step(action)[:4]

    @tf.function
    def step(self, action):
        """Create an in-graph operations that updates a gym.Env with
        actions sampled from an agent

        Args:

        action: tf.dtypes.float32
            a tensor that represents a single action sampled from an agent
        """

        obs, ctx, r, d = tf.py_function(self._step, [
            action], [tf.float32, tf.float32, tf.float32, tf.bool])
        obs.set_shape(self.observation_space.shape)
        ctx.set_shape(self.context_space.shape)
        r.set_shape(tf.TensorShape([1]))
        d.set_shape(tf.TensorShape([1]))
        return obs, ctx, r, d

    def _reset(self):
        return self.wrapped_env.reset()

    @tf.function
    def reset(self):
        """Create an in-graph operations that resets a gym.Env and returns
        the initial observation
        """

        obs, ctx = tf.py_function(self._reset, [], [tf.float32, tf.float32])
        obs.set_shape(self.observation_space.shape)
        ctx.set_shape(self.context_space.shape)
        return obs, ctx

    def _render(self):
        return self.wrapped_env.render(mode='rgb_array').astype(float)

    @tf.function
    def render(self):
        """Create an in-graph operations that renders a gym.Env and returns
        the image pixels in a tensor
        """
        return tf.py_function(self._render, [], tf.float32)
