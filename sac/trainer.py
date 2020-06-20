import tensorflow as tf


class Trainer(object):

    def __init__(self,
                 training_env,
                 eval_env,
                 policy,
                 buffer,
                 algorithm,
                 logger,
                 warm_up_steps=5000,
                 evaluate_every=5000,
                 batch_size=256):
        """Create a training interface for an rl agent using
        the provided rl algorithm

        Args:

        training_env: Env
            a gym environment wrapped in a static graph interface
        eval_env: Env
            a gym environment wrapped in a static graph interface
        policy: tf.keras.Model
            a neural network that returns an action probability distribution
        buffer: ReplayBuffer
            a static graph replay buffer for sampling batches of data
        algorithm: list of Algorithms
            an rl algorithm that takes in batches of data in a train function
        logger: Logger
            an intergace that supports a record function for tensorboard
        """

        self.training_env = training_env
        self.eval_env = eval_env
        self.policy = policy
        self.buffer = buffer
        self.algorithm = algorithm
        self.logger = logger
        self.warm_up_steps = warm_up_steps
        self.evaluate_every = evaluate_every
        self.batch_size = batch_size
        self.obs = tf.Variable(self.training_env.reset())

    @tf.function
    def evaluate(self, num_paths):
        """Evaluate the current policy by collecting data over many episodes
        and returning the sum of rewards

        Args:

        num_paths: tf.dtypes.int32
            the number of episodes to collect when evaluating
        """

        array = tf.TensorArray(tf.dtypes.float32, size=num_paths)
        path_return = tf.constant([0.0])
        obs = self.eval_env.reset()
        i = tf.constant(0)
        while tf.less(i, num_paths):
            act = self.policy.get_distribution(
                self.policy([obs[tf.newaxis]])).mean()[0]
            obs, reward, done = self.eval_env.step(act)
            path_return += reward
            if done:
                array = array.write(i, path_return)
                path_return = tf.constant([0.0])
                obs = self.eval_env.reset()
                i = i + 1
        return array.stack()

    @tf.function
    def train(self, i):
        """Train the current policy by collecting data over many episodes
        and running the provided rl algorithm

        Args:

        i: tf.dtypes.int64
            the training iteration or the number of steps collected
        """

        i = tf.cast(i, tf.dtypes.int64)
        if tf.greater_equal(i, self.warm_up_steps):
            act = self.policy.get_distribution(
                self.policy([self.obs[tf.newaxis]])).sample()[0]
            self.algorithm.train(i, *self.buffer.sample(self.batch_size))
        else:
            act = self.training_env.action_space.sample()
        next_obs, reward, done = self.training_env.step(act)
        self.buffer.insert(self.obs, act, reward, done)
        if done:
            self.obs.assign(self.training_env.reset())
        else:
            self.obs.assign(next_obs)
        if i % self.evaluate_every == 0:
            self.logger.record("return", self.evaluate(10), i)
