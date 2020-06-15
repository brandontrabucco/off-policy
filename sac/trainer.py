from skvideo.io import vwrite
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

    def _save_vid(self, name, vid):
        print("saving", name)
        vwrite(name.decode("utf-8"), vid)

    def save_vid(self, name, vid):
        tf.numpy_function(self._save_vid, [name, vid], [])

    def render(self, number):
        """Evaluate the current policy by collecting data over many episodes
        and returning the sum of rewards

        Args:

        num_paths: tf.dtypes.int32
            the number of episodes to collect when evaluating
        """

        obs, ctx = self.env.reset()
        array = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        i = tf.constant(0)
        done = tf.constant([False])
        array = array.write(i, self.env.render() * 255)

        while tf.logical_not(done):
            i = i + 1

            act = self.policy([
                obs[tf.newaxis], ctx[tf.newaxis]]).sample()[0]

            obs, ctx, reward, done = self.env.step(act)
            array = array.write(i, self.env.render() * 255)
            i = i + 1
        self.save_vid(tf.constant("a.wmv"), array.stack())

    def evaluate(self, num_paths):
        """Evaluate the current policy by collecting data over many episodes
        and returning the sum of rewards

        Args:

        num_paths: tf.dtypes.int32
            the number of episodes to collect when evaluating
        """

        obs, ctx = self.env.reset()
        array = tf.TensorArray(tf.float32, size=num_paths)

        i = tf.constant(0)
        path_return = tf.constant([0.0])

        while tf.less(i, num_paths):

            act = self.policy([
                obs[tf.newaxis], ctx[tf.newaxis]]).sample()[0]

            obs, ctx, reward, done = self.env.step(act)
            path_return += reward

            if done:
                obs, ctx = self.env.reset()
                array = array.write(i, path_return)

                i = i + 1
                path_return = tf.constant([0.0])

        return array.stack()

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
                #self.render(i)
