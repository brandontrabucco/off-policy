import tensorflow as tf


class Trainer(object):

    def __init__(self, training_env, eval_env, policy, buffer, sac,
                 episodes_per_eval=10, warm_up_steps=5000, batch_size=256,
                 variance_scaling=10.0, clip_range=2.0):
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
        algorithm: Algorithm
            an rl algorithm that takes in batches of data in a train function
        episodes_per_eval: int
            the number of episodes to collect when evaluating the policy
        warm_up_steps: int
            the minimum number of steps to collect before training the policy
        batch_size: int
            the number of samples per batch to use when training the policy
        variance_scale: float
            multiplied onto std of observations during normalization
        clip_range: float
            range of normalized observations to clip values to stay within
        """

        # create the training machinery for an off policy algorithm
        self.training_env = training_env
        self.eval_env = eval_env
        self.policy = policy
        self.buffer = buffer
        self.algorithm = sac

        # set hyper parameters for the off policy algorithm
        self.episodes_per_eval = episodes_per_eval
        self.warm_up_steps = warm_up_steps
        self.batch_size = batch_size

        # set hyper parameters for normalization
        self.variance_scaling = variance_scaling
        self.clip_range = clip_range

        # reset the environment and track the previous observation
        self.obs = tf.Variable(self.process_obs(
            self.training_env.reset()), dtype=tf.float32)

    @tf.function
    def process_obs(self, obs, batched=False):
        """Process observations by normalizing them using statistics
        calculated from the replay buffer and filtering for outliers

        Args:

        obs: tf.Tensor
            an input tensor of observations from the environment

        Returns:

        normalized_obs: tf.Tensor
            a normalized input tensor of observations
        """

        # check if enough observations have been collected
        active = tf.greater_equal(
            self.buffer.step, self.warm_up_steps)

        if batched and active:
            # normalize the observations with a mean and variance
            obs = obs - self.buffer.obs_shift[tf.newaxis, :]
            obs = obs / self.buffer.obs_scale[tf.newaxis, :]

            if self.clip_range is not None:
                # clip the observations to remove outliers
                return tf.clip_by_value(
                    obs, -self.clip_range, self.clip_range)

            else:
                # observation clipping is disabled
                return obs

        elif active:
            # normalize the observations with a mean and variance
            obs = obs - self.buffer.obs_shift
            obs = obs / self.buffer.obs_scale

            if self.clip_range is not None:
                # clip the observations to remove outliers
                return tf.clip_by_value(
                    obs, -self.clip_range, self.clip_range)

            else:
                # observation clipping is disabled
                return obs

        else:
            # normalization is not active yet
            return obs

    @tf.function
    def train(self):
        """Train the current policy by collecting data over many episodes
        and running the provided rl algorithm
        """

        if tf.equal(self.buffer.step, self.warm_up_steps):
            # slice out the collected observations from the buffer
            obs_slice = self.buffer.obs[:self.warm_up_steps + 1]

            # compute the observation statistics from the buffer
            shift = tf.reduce_mean(obs_slice, 0)
            self.buffer.obs_shift.assign(shift)

            # assign normalization parameters of the environments
            scale = tf.math.reduce_std(obs_slice - shift[tf.newaxis, :], 0)
            scale = tf.clip_by_value(scale, 1e-6, 1e9) * self.variance_scaling
            self.buffer.obs_scale.assign(scale)

            # rescale the current observation from the environment
            self.obs.assign(self.process_obs(self.obs))

            # rescale the observations in the buffer
            obs_slice = self.process_obs(obs_slice, batched=True)
            self.buffer.obs.assign(tf.pad(obs_slice, [[
                0, self.buffer.capacity - self.warm_up_steps - 1], [0, 0]]))

        if tf.greater_equal(self.buffer.step, self.warm_up_steps):
            # train the policy using an off policy algorithm
            self.algorithm.train(*self.buffer.sample(self.batch_size))

            # sample actions from the current policy
            act = self.policy(self.obs[tf.newaxis]).sample()[0]

        else:
            # sample actions randomly
            act = self.training_env.action_space.sample()

        # step the environment by taking an action
        next_obs, reward, done, info = self.training_env.step(act)

        # insert collected samples into the replay buffer
        self.buffer.insert(self.obs, act, reward, done)

        # if the end of an episode is reached then reset environment
        if done:
            next_obs = self.training_env.reset()

        # prepare the observation for the next iteration
        self.obs.assign(self.process_obs(next_obs))

    @tf.function
    def evaluate(self, num_paths):
        """Evaluate the current policy by collecting data over many episodes
        and returning the sum of rewards

        Args:

        num_paths: tf.dtypes.int32
            the number of episodes to collect when evaluating

        Returns:

        returns: tf.dtypes.float32
            a tensor containing returns from num_paths independent trials
        lengths: tf.dtypes.float32
            a tensor containing the length of episodes that were sampled
        """

        # track both the return and episode length
        return_array = tf.TensorArray(tf.dtypes.float32, size=num_paths)
        length_array = tf.TensorArray(tf.dtypes.float32, size=num_paths)

        # collect num_paths episodes in total
        for i in tf.range(num_paths):
            # initialize the evaluation environment
            obs, done = self.eval_env.reset(), tf.constant([False])
            returns = tf.constant([0.0])
            lengths = tf.constant([0.0])

            # run the episode until termination
            while tf.logical_not(done):
                # take the mean action of the policy
                obs = self.process_obs(obs)
                act = self.policy(obs[tf.newaxis]).mean()[0]

                # step the environment by taking an action
                obs, reward, done, info = self.eval_env.step(act)

                # update the return and episode length
                returns += reward
                lengths += 1.0

            # store the return and length attained in the current trial
            return_array = return_array.write(i, returns)
            length_array = length_array.write(i, lengths)

        # return the return and length as tensors
        return return_array.stack(), length_array.stack()

    @tf.function
    def get_diagnostics(self):
        """Gather diagnostic information from the learning algorithm,
        and return a dict containing tensors

        Returns:

        diagnostics: dict
            a dict containing tensors whose statistics will be summarized
        """

        # evaluate the current policy
        returns, lengths = self.evaluate(self.episodes_per_eval)

        # also get diagnostic information from the algorithm
        return {"evaluate/return": returns, "evaluate/length": lengths,
                **self.algorithm.get_diagnostics(
                    *self.buffer.sample(self.batch_size))}
