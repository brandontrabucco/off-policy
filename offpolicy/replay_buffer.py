import tensorflow as tf


class ReplayBuffer(tf.Module):

    def __init__(self, capacity: int, obs_size: int, act_size: int):
        """A static graph replay buffer that stores samples collected
        from a policy in an environment

        Args:

        capacity: int
            the number of samples that can be in the buffer, maximum
        obs_size: int
            the number of channels in the observation space, flattened
        act_size: int
            the number of channels in the action space, flattened
        """

        super(ReplayBuffer, self).__init__()
        self.capacity = capacity
        self.obs_size = obs_size
        self.act_size = act_size

        # prepare a storage memory for samples
        self.obs = tf.Variable(
            tf.zeros([capacity, obs_size], tf.dtypes.float32))
        self.act = tf.Variable(
            tf.zeros([capacity, act_size], tf.dtypes.float32))
        self.reward = tf.Variable(
            tf.zeros([capacity, 1], tf.dtypes.float32))
        self.done = tf.Variable(
            tf.zeros([capacity, 1], tf.dtypes.bool))

        # save size statistics for the buffer
        self.head = tf.Variable(tf.constant(0))
        self.size = tf.Variable(tf.constant(0))
        self.step = tf.Variable(tf.constant(-1))

        # variables that will be used frequently during training
        self.obs_range = tf.reshape(
            tf.range(self.obs_size), [self.obs_size, 1])
        self.act_range = tf.reshape(
            tf.range(self.act_size), [self.act_size, 1])

        # set the initial normalizer values
        self.obs_shift = tf.Variable(
            tf.zeros([obs_size], tf.float32))
        self.obs_scale = tf.Variable(
            tf.ones([obs_size], tf.float32))

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
        location = tf.concat([tf.tile(tf.reshape(
            self.head, [1, 1]), [self.obs_size, 1]), self.obs_range], 1)
        self.obs.assign(tf.tensor_scatter_nd_update(
            self.obs, location, tf.cast(obs, tf.float32)))

        # insert samples at the position of the current head
        location = tf.concat([tf.tile(tf.reshape(
            self.head, [1, 1]), [self.act_size, 1]), self.act_range], 1)
        self.act.assign(tf.tensor_scatter_nd_update(
            self.act, location, tf.cast(act, tf.float32)))

        # insert samples at the position of the current head
        location = tf.pad(tf.reshape(self.head, [1, 1]), [[0, 0], [0, 1]])
        self.reward.assign(tf.tensor_scatter_nd_update(
            self.reward, location, tf.cast(reward, tf.float32)))

        # insert samples at the position of the current head
        location = tf.pad(tf.reshape(self.head, [1, 1]), [[0, 0], [0, 1]])
        self.done.assign(tf.tensor_scatter_nd_update(
            self.done, location, tf.cast(done, tf.bool)))

        # increment the size statistics of the buffer
        self.head.assign(tf.math.floormod(self.head + 1, self.capacity))
        self.size.assign(tf.minimum(self.size + 1, self.capacity))
        self.step.assign(self.step + 1)

    @tf.function
    def sample(self, batch_size):
        """Samples a batch of training data from the replay buffer
        and returns the batch of data

        Args:

        batch_size: tf.dtypes.int32
            a scalar tensor that specifies how many elements to sample

        Returns:

        step: tf.dtypes.int64
            a scalar tensor that indicates the training iteration
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

        indices = tf.random.uniform([
            batch_size], maxval=self.size, dtype=tf.dtypes.int32)
        next_indices = tf.math.floormod(indices + 1, self.capacity)
        return (self.step, tf.gather(self.obs, indices, axis=0),
                tf.gather(self.act, indices, axis=0),
                tf.gather(self.reward, indices, axis=0),
                tf.gather(self.done, indices, axis=0),
                tf.gather(self.obs, next_indices, axis=0))
