import tensorflow as tf


class ReplayBuffer(object):

    def __init__(self, capacity, obs_size, act_size):
        """A static graph replay buffer that stores samples collected
        from a policy in an environment

        Args:

        capacity: tf.dtypes.int32
            the number of samples that can be in the buffer, maximum
        obs_size: tf.dtypes.int32
            the number of channels in the observation space, flattened
        act_size: tf.dtypes.int32
            the number of channels in the action space, flattened
        """

        super(ReplayBuffer, self).__init__()
        self.capacity = capacity
        self.obs_size = obs_size
        self.act_size = act_size

        # prepare a storage memory for samples
        self.obs = tf.Variable(tf.zeros([capacity, obs_size], tf.dtypes.float32))
        self.act = tf.Variable(tf.zeros([capacity, act_size], tf.dtypes.float32))
        self.reward = tf.Variable(tf.zeros([capacity, 1], tf.dtypes.float32))
        self.done = tf.Variable(tf.zeros([capacity, 1], tf.dtypes.bool))

        # save size statistics for the buffer
        self.head = tf.Variable(tf.constant(0))
        self.size = tf.Variable(tf.constant(0))

    @tf.function
    def insert(self, obs, act, reward, done):
        """Insert a single sample collected from the environment into
        the replay buffer at the current head position

        Args:

        obs: tf.dtypes.float32
            a tensor that is shaped like [obs_size] containing observations
        act: tf.dtypes.float32
            a tensor that is shaped like [act_size] containing actions
        reward: tf.dtypes.float32
            a tensor that is shaped like [1] containing a reward
        done: tf.dtypes.bool
            a tensor that is shaped like [1] containing a done signal
        """

        # insert samples at the position of the current head
        location = tf.concat([
            tf.tile(tf.reshape(self.head, [1, 1]), [self.obs_size, 1]),
            tf.reshape(tf.range(self.obs_size), [self.obs_size, 1])], 1)
        self.obs.assign(tf.tensor_scatter_nd_update(
            self.obs, location, tf.cast(obs, tf.dtypes.float32)))

        # insert samples at the position of the current head
        location = tf.concat([
            tf.tile(tf.reshape(self.head, [1, 1]), [self.act_size, 1]),
            tf.reshape(tf.range(self.act_size), [self.act_size, 1])], 1)
        self.act.assign(tf.tensor_scatter_nd_update(
            self.act, location, tf.cast(act, tf.dtypes.float32)))

        # insert samples at the position of the current head
        location = tf.pad(tf.reshape(self.head, [1, 1]), [[0, 0], [0, 1]])
        self.reward.assign(tf.tensor_scatter_nd_update(
            self.reward, location,
            tf.cast(tf.reshape(reward, [1]), tf.dtypes.float32)))

        # insert samples at the position of the current head
        location = tf.pad(tf.reshape(self.head, [1, 1]), [[0, 0], [0, 1]])
        self.done.assign(tf.tensor_scatter_nd_update(
            self.done, location,
            tf.cast(tf.reshape(done, [1]), tf.dtypes.bool)))

        # increment the size statistics of the buffer
        self.head.assign(tf.math.floormod(self.head + 1, self.capacity))
        self.size.assign(tf.minimum(self.size + 1, self.capacity))

    @tf.function
    def sample(self, batch_size):
        """Samples a batch of training data from the replay buffer
        and returns the batch of data

        Args:

        batch_size: tf.dtypes.int32
            a scalar tensor that specifies how many elements to sample

        Returns:

        obs: tf.dtypes.float32
            a tensor shaped [batch_size, obs_size] containing observations
        act: tf.dtypes.float32
            a tensor shaped [batch_size, act_size] containing actions
        reward: tf.dtypes.float32
            a tensor shaped [batch_size, 1] containing a reward
        done: tf.dtypes.bool
            a tensor shaped [batch_size, 1] containing a done signal
        next_obs: tf.dtypes.float32
            a tensor shaped [batch_size, obs_size] containing observations
        """

        indices = tf.random.shuffle(tf.range(self.size))[:batch_size]
        next_indices = tf.math.floormod(indices + 1, self.capacity)
        return (tf.gather(self.obs, indices, axis=0),
                tf.gather(self.act, indices, axis=0),
                tf.gather(self.reward, indices, axis=0),
                tf.gather(self.done, indices, axis=0),
                tf.gather(self.obs, next_indices, axis=0))
