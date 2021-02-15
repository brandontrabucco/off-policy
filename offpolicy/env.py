import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import gym


class StaticGraphBox(gym.spaces.Space):

    def __init__(self, low: tf.Tensor, high: tf.Tensor):
        """Create a space that supports TensorFlow operations
        and sampling TensorFlow tensors

        Args:

        low: tf.dtypes.float32
            a tensor that specifies the lower bound for the space
        high: tf.dtypes.float32
            a tensor that specifies the upper bound for the space
        """

        super(StaticGraphBox, self).__init__(None, None)
        self.low = tf.convert_to_tensor(low, dtype=tf.float32)
        self.high = tf.convert_to_tensor(high, dtype=tf.float32)
        self.shape = self.low.get_shape()
        self.dtype = self.low.dtype

        # check if the space is upper and lower bounded
        self.lb = tf.logical_not(tf.math.is_nan(self.low))
        self.ub = tf.logical_not(tf.math.is_nan(self.high))

        # create sampling distributions for the space
        self.exp = tfp.distributions.Exponential(tf.ones_like(self.low))
        self.uniform = tfp.distributions.Uniform(self.low, self.high)
        self.normal = tfp.distributions.Normal(
            tf.zeros_like(self.low), tf.ones_like(self.low))

    @tf.function
    def sample(self):
        """Draw samples from a probability distribution whose support is
        the entire space defined by low and high
        """

        normal, exp = self.normal.sample(), self.exp.sample()
        uniform = self.uniform.sample()
        samples = tf.where(self.lb, self.low + exp, normal)
        samples = tf.where(self.ub, self.high - exp, samples)
        return tf.where(tf.logical_and(
            self.lb, self.ub), uniform, samples)

    @tf.function
    def contains(self, x):
        """Check if a TensorFlow tensor is contained within the upper
        and lower bounds of this space

        Args:

        x: tf.dtypes.float32
            a tensor that shall be compared with the upper and lower bounds
        """

        return tf.logical_and(tf.greater_equal(x, self.low),
                              tf.less_equal(x, self.high))


class StaticGraphEnv(gym.Env):

    def __init__(self, env: gym.Env,
                 info_names: tuple = (), info_shape: tuple = ()):
        """Create an in-graph environment with the same API as the gym.Env
        class, but with static graph operations

        Args:

        wrapped_env: gym.Env
            an OpenAI Gym environment with step, reset, and render functions
        info_names: tuple[str]
            a list of environment info keys to convert to TensorFlow
        info_shape: tuple[list[int]]
            a list of environment info shapes corresponding to the keys
        """

        self.env = env
        self.reward_range = tf.convert_to_tensor(env.reward_range)
        self.observation_space = StaticGraphBox(
            env.observation_space.low, env.observation_space.high)
        self.action_space = StaticGraphBox(
            env.action_space.low, env.action_space.high)

        # create a buffer for storing intermediate values
        self.obs = self.reward = self.done = self.info = None
        self.info_names = info_names
        self.info_shape = info_shape
        self.info_types = [tf.float32] * len(self.info_names)

    def buffered_step(self, action):
        """Perform a step with the wrapped gym environment and store
        observations, rewards, and env info locally

        Args:

        action: np.float32
            a tensor that represents a single action sampled from an agent
        """

        self.obs, self.reward, self.done, self.info = self.env.step(action)

    def get_data(self):
        """Convert observations, rewards, and env info into a format that
        can be returned from a TensorFlow static graph function

        Returns:

        obs: np.float32
            an array representing the observation in the latest time step
        reward: np.float32
            an array representing the reward attained in the latest time step
        done: np.float32
            an array representing when the episode reaches termination
        info: list[np.float32]
            an list of arrays representing information from the environment
        """

        return [self.obs.astype(np.float32),
                np.reshape(self.reward, [1]).astype(np.float32),
                np.reshape(self.done, [1]).astype(np.bool)] + [
                   np.reshape(self.info[name],
                              self.info_shape[idx]).astype(np.float32)
                   for idx, name in enumerate(self.info_names)]

    def convert_to_gym(self, obs, reward, done, info):
        """Convert observations, rewards, and env info into a format that
        can be returned from a TensorFlow static graph function

        Args:

        obs: tf.float32
            a tensor representing the observation in the latest time step
        reward: tf.float32
            a tensor representing the reward attained in the latest time step
        done: tf.float32
            a tensor representing when the episode reaches termination
        info: list[tf.float32]
            a list of tensors representing information from the environment

        Returns:

        obs: tf.float32
            a tensor representing the observation in the latest time step
        reward: tf.float32
            a tensor representing the reward attained in the latest time step
        done: tf.float32
            a tensor representing when the episode reaches termination
        info: dict[str, tf.float32]
            a dict of tensors representing information from the environment
        """

        # first format the obs and rewards
        obs.set_shape(self.observation_space.shape)
        reward.set_shape(tf.TensorShape([1]))
        done.set_shape(tf.TensorShape([1]))

        # second format the env info
        info_dict = dict()
        for idx, info_value in enumerate(info):
            info_value.set_shape(tf.TensorShape(self.info_shape[idx]))
            info_dict[self.info_names[idx]] = info_value
        return obs, reward, done, info_dict

    @tf.function
    def step(self, action):
        """Create an in-graph operation that updates a gym.Env with
        actions sampled from an agent and return the result

        Args:

        action: tf.dtypes.float32
            a tensor that represents a single action sampled from an agent

        Returns:

        obs: tf.float32
            a tensor representing the observation in the latest time step
        reward: tf.float32
            a tensor representing the reward attained in the latest time step
        done: tf.float32
            a tensor representing when the episode reaches termination
        info: dict[str, tf.float32]
            a dict of tensors representing information from the environment
        """

        action = tf.where(tf.math.is_nan(action), tf.zeros_like(action), action)
        with tf.control_dependencies([
                tf.numpy_function(self.buffered_step, [action], [])]):
            obs, reward, done, *info = tf.numpy_function(
                self.get_data, [], [
                    tf.float32, tf.float32, tf.bool] + self.info_types)
            return self.convert_to_gym(obs, reward, done, info)

    def _reset(self):
        """Create an in-graph operation that resets a gym.Env and returns
        the initial observation
        """

        return self.env.reset().astype(np.float32)

    @tf.function
    def reset(self):
        """Create an in-graph operation that resets a gym.Env and returns
        the initial observation
        """

        obs = tf.numpy_function(self._reset, [], tf.float32)
        obs.set_shape(self.observation_space.shape)
        return obs

    def _render(self):
        """Create an in-graph operation that renders a gym.Env and displays
        the rendered environment to the screen
        """

        return self.env.render(mode='human')

    @tf.function
    def render(self):
        """Create an in-graph operation that renders a gym.Env and displays
        the rendered environment to the screen
        """

        tf.numpy_function(self._render, [], [])
