import tensorflow as tf


class Trainer(object):

    def __init__(self,
                 env,
                 policy,
                 buffer,
                 algorithm,
                 logger,
                 video_saver,
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
        self.video_saver = video_saver
        self.warm_up_steps = warm_up_steps
        self.batch_size = batch_size

    @tf.function
    def render(self, number):
        """Evaluate the current policy by collecting data over many episodes
        and returning the sum of rewards

        Args:

        num_paths: tf.dtypes.int32
            the number of episodes to collect when evaluating
        """

        obs, ctx = self.env.reset()
        done = tf.constant([False])
        self.video_saver.open(number)
        while tf.logical_not(done):
            act = self.policy([
                obs[tf.newaxis], ctx[tf.newaxis]]).mean()[0]
            obs, ctx, reward, done = self.env.step(act)
            self.video_saver.write_frame(self.env.render())
        self.video_saver.close()

    @tf.function
    def evaluate(self, num_paths):
        """Evaluate the current policy by collecting data over many episodes
        and returning the sum of rewards

        Args:

        num_paths: tf.dtypes.int32
            the number of episodes to collect when evaluating
        """

        obs, ctx = self.env.reset()
        array = tf.TensorArray(tf.float32, size=num_paths)
        video = tf.TensorArray(tf.float32, size=num_paths)

        i = tf.constant(0)
        path_return = tf.constant([0.0])

        while tf.less(i, num_paths):

            act = self.policy([
                obs[tf.newaxis], ctx[tf.newaxis]]).mean()[0]

            obs, ctx, reward, done = self.env.step(act)
            video = video.write(i, self.env.render())
            path_return += reward

            if done:
                obs, ctx = self.env.reset()
                array = array.write(i, path_return)

                i = i + 1
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

        obs, ctx = self.env.reset()
        i = tf.constant(0)

        while tf.less(i, num_iterations):

            i = i + 1
            self.logger.set_step(tf.cast(i, tf.dtypes.int64))
            if tf.greater(i, self.warm_up_steps):

                act = self.policy([
                    obs[tf.newaxis], ctx[tf.newaxis]]).sample()[0]

                (obs_s, ctx_s, act_s, reward_s, done_s, next_obs_s,
                 next_ctx_s) = self.buffer.sample(self.batch_size)

                self.algorithm.train(
                    i, tf.concat([obs_s, ctx_s], axis=-1), act_s, reward_s,
                    done_s, tf.concat([next_obs_s, next_ctx_s], axis=-1))

            else:
                act = self.env.action_space.sample()

            next_obs, next_ctx, reward, done = self.env.step(act)
            self.buffer.insert(obs, ctx, act, reward, done)

            obs, ctx = next_obs, next_ctx
            if done:
                obs, ctx = self.env.reset()

            if i % 1000 == 0:
                self.logger.set_step(tf.cast(i, tf.dtypes.int64))
                self.logger.record("return", self.evaluate(10))
                self.render(i)
