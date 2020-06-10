import tensorflow as tf


class Trainer(object):

    def __init__(self,
                 env,
                 policy,
                 buffer,
                 algorithm,
                 logger,
                 warm_up_steps=5000,
                 batch_size=256):
        """Create a training interface for an rl agent using
        the provided rl algorithm

        Args:

        env: Env
            an openai gym environment wrapped in a static graph interface
        policy: tf.keras.Model
            a neural network that returns an action probability distribution
        buffer: ReplayBuffer
            a static graph replay buffer for sampling batches of data
        algorithm: SAC
            an rl algorithm that takes in batches of data in a train function
        logger: Logger
            an intergace that supports a record function for tensorboard
        """

        self.env = env
        self.policy = policy
        self.buffer = buffer
        self.algorithm = algorithm
        self.logger = logger
        self.warm_up_steps = warm_up_steps
        self.batch_size = batch_size

    @tf.function
    def evaluate(self, num_paths):
        """Evaluate the current policy by collecting data over many episodes
        and returning the sum of rewards

        Args:

        num_paths: tf.dtypes.int32
            the number of episodes to collect when evaluating
        """

        obs = self.env.reset()
        array = tf.TensorArray(tf.float32, size=num_paths)
        paths = tf.constant(0)
        path_return = tf.constant([0.0])

        while tf.less(paths, num_paths):

            act = self.policy([obs[tf.newaxis]]).mean()[0]
            obs, reward, done = self.env.step(act)
            path_return += reward

            if done:
                obs = self.env.reset()
                array = array.write(paths, path_return)
                paths = paths + 1
                path_return = tf.constant([0.0])

        return array.stack()

    @tf.function
    def train(self, num_iterations):
        """Train the current policy by collecting data over many episodes
        and running the provided rl algorithm

        Args:

        num_iterations: tf.dtypes.int32
            the number of steps to collect when training
        """

        iteration = tf.constant(0)
        obs = self.env.reset()

        while tf.less(iteration, num_iterations):

            iteration = iteration + 1
            self.logger.set_step(tf.cast(iteration, tf.dtypes.int64))

            if tf.greater(iteration, self.warm_up_steps):
                act = self.policy([obs[tf.newaxis]]).sample()[0]
                self.algorithm.train(
                    iteration, *self.buffer.sample(self.batch_size))
            else:
                act = self.env.action_space.sample()

            next_obs, reward, done = self.env.step(act)
            self.buffer.insert(obs, act, reward, done)
            obs = self.env.reset() if done else next_obs

            if iteration % 1000 == 0:
                self.logger.set_step(tf.cast(iteration, tf.dtypes.int64))
                self.logger.record("return", self.evaluate(10))
