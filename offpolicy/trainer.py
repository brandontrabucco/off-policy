import tensorflow as tf


class Trainer(object):

    def __init__(self,
                 training_env,
                 eval_env,
                 policy,
                 buffer,
                 sac,
                 normalized_obs=True,
                 normalizer_tau=tf.constant(5e-3),
                 episodes_per_eval=10,
                 warm_up_steps=5000,
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
        algorithm: Algorithm
            an rl algorithm that takes in batches of data in a train function
        """

        self.training_env = training_env
        self.eval_env = eval_env
        self.policy = policy
        self.buffer = buffer
        self.sac = sac
        self.normalized_obs = normalized_obs
        self.normalizer_tau = normalizer_tau
        self.episodes_per_eval = episodes_per_eval
        self.warm_up_steps = warm_up_steps
        self.batch_size = batch_size
        self.obs = tf.Variable(
            self.training_env.reset(), dtype=tf.float32)
        self.running_mean = tf.Variable(
            tf.zeros_like(self.obs[tf.newaxis]), dtype=tf.float32)
        self.running_std = tf.Variable(
            tf.ones_like(self.obs[tf.newaxis]), dtype=tf.float32)

    @tf.function
    def n(self, x):
        """Normalize an observation using the running normalization statistics
        calculated from samples in the replay buffer

        Args:

        x: tf.dtypes.float32
            a tensor containing observations that might be normalized here
        """

        if self.normalized_obs:
            x = tf.math.divide_no_nan(x - self.running_mean, self.running_std)
        return x

    @tf.function
    def update_normalizer(self, tau):
        """Update the normalization statistics used for normalizing
        observations provided to the policy

        Args:

        tau: tf.dtypes.float32
            a parameter to control the extent of the target update
        """

        if self.normalized_obs:
            mean = tf.math.reduce_mean(
                self.buffer.obs, axis=0, keepdims=True)
            std = tf.math.reduce_std(
                self.buffer.obs - mean, axis=0, keepdims=True)
            self.running_mean.assign(
                tau * mean + (1.0 - tau) * self.running_mean)
            self.running_std.assign(
                tau * std + (1.0 - tau) * self.running_std)

    @tf.function
    def train(self):
        """Train the current policy by collecting data over many episodes
        and running the provided rl algorithm
        """

        if tf.greater_equal(self.buffer.step, self.warm_up_steps):
            i, obs, act, r, d, next_obs = self.buffer.sample(self.batch_size)
            self.sac.train(i, self.n(obs), act, r, d, self.n(next_obs))
            act = self.policy.sample([self.n(self.obs[tf.newaxis])])[0]
            self.update_normalizer(self.normalizer_tau)
        else:
            act = self.training_env.action_space.sample()
            self.update_normalizer(tf.constant(1.0))
        next_obs, reward, done = self.training_env.step(act)
        self.buffer.insert(self.obs, act, reward, done)
        if done:
            next_obs = self.training_env.reset()
        self.obs.assign(next_obs)

    @tf.function
    def evaluate(self,
                 num_paths):
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

        return_array = tf.TensorArray(tf.dtypes.float32, size=num_paths)
        length_array = tf.TensorArray(tf.dtypes.float32, size=num_paths)
        for i in tf.range(num_paths):
            obs, done = self.eval_env.reset(), tf.constant([False])
            returns = tf.constant([0.0])
            lengths = tf.constant([0.0])
            while tf.logical_not(done):
                act = self.policy.mean([self.n(obs[tf.newaxis])])[0]
                obs, rew, done = self.eval_env.step(act)
                returns += rew
                lengths += 1.0
            return_array = return_array.write(i, returns)
            length_array = length_array.write(i, lengths)
        return return_array.stack(), length_array.stack()

    @tf.function
    def get_diagnostics(self):
        """Gather diagnostic information from the learning algorithm,
        and return a dict containing tensors

        Returns:

        diagnostics: dict
            a dict containing tensors whose statistics will be summarized
        """

        returns, lengths = self.evaluate(self.episodes_per_eval)
        i, obs, act, r, d, next_obs = self.buffer.sample(self.batch_size)
        return {"return": returns, "length": lengths,
                "running_mean": self.running_mean,
                "running_std": self.running_std,
                **self.sac.get_diagnostics(
                    i, self.n(obs), act, r, d, self.n(next_obs))}

    def get_saveables(self):
        """Collects and returns stateful objects that are serializeable
        using the tensorflow checkpoint format

        Returns:

        saveables: dict
            a dict containing stateful objects compatible with checkpoints
        """

        return {"buffer": self.buffer, "policy": self.policy,
                "running_mean": self.running_mean,
                "running_std": self.running_std,
                **self.sac.get_saveables()}
