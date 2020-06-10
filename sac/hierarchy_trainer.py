import tensorflow as tf


class HierarchyTrainer(object):

    def __init__(self,
                 env,
                 policies,
                 buffers,
                 algorithms,
                 logger,
                 period=10,
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
        self.policies = policies
        self.buffers = buffers
        self.algorithms = algorithms
        self.logger = logger
        self.period = period
        self.warm_up_steps = warm_up_steps
        self.batch_size = batch_size
        self.levels = len(self.policies)

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

        i = tf.constant(0)
        time_step = tf.constant(0)
        path_return = tf.constant([0.0])

        obs_i = [None for p in self.policies]
        act_i = [None for p in self.policies]

        while tf.less(i, num_paths):

            for level, p in enumerate(self.policies):
                if time_step % self.period ** (
                        self.levels - level - 1) == 0:
                    if level > 0:
                        ctx = act_i[level - 1] - (obs - obs_i[level - 1])
                    obs_i[level] = obs
                    act_i[level] = p([obs[tf.newaxis],
                                      ctx[tf.newaxis]]).mean()[0]

            next_obs, next_ctx, reward, done = self.env.step(act_i[-1])
            time_step = time_step + 1
            path_return += reward

            obs, ctx = next_obs, next_ctx
            if done:
                obs, ctx = self.env.reset()
                array = array.write(i, path_return)

                i = i + 1
                time_step = tf.constant(0)
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
        time_step = tf.constant(0)

        reward_i = [None for p in self.policies]
        done_i = [None for p in self.policies]
        obs_i = [None for p in self.policies]
        ctx_i = [None for p in self.policies]
        act_i = [None for p in self.policies]

        while tf.less(i, num_iterations):

            i = i + 1
            self.logger.set_step(tf.cast(i, tf.dtypes.int64))

            for level, (p, b, a) in enumerate(zip(self.policies,
                                                  self.buffers,
                                                  self.algorithms)):
                if time_step % self.period ** (
                        self.levels - level - 1) == 0:

                    if time_step > 0:
                        b.insert(obs_i[level],
                                 ctx_i[level],
                                 act_i[level],
                                 reward_i[level],
                                 done_i[level])

                    if level > 0:
                        ctx = act_i[level - 1] - (obs - obs_i[level - 1])
                    reward_i[level] = tf.constant([0.0])
                    done_i[level] = tf.constant([False])
                    obs_i[level] = obs
                    ctx_i[level] = ctx
                    act_i[level] = p([obs[tf.newaxis],
                                      ctx[tf.newaxis]]).sample()[0]

                if i > self.warm_up_steps:
                    obs_s, ctx_s, *rest = b.sample(self.batch_size)
                    a.train(i, tf.concat([obs_s, ctx_s], axis=-1), *rest)

            next_obs, next_ctx, reward, done = self.env.step(act_i[-1])
            time_step = time_step + 1

            for level, p in enumerate(self.policies):
                reward_i[level] = reward_i[level] + reward
                done_i[level] = done

            obs, ctx = next_obs, next_ctx
            if done:
                obs, ctx = self.env.reset()
                time_step = 0
                self.buffers[level].insert(obs_i[level],
                                           ctx_i[level],
                                           act_i[level],
                                           reward_i[level],
                                           done_i[level])

            if i % 1000 == 0:
                self.logger.set_step(tf.cast(i, tf.dtypes.int64))
                self.logger.record("return", self.evaluate(10))
