import tensorflow as tf
import tensorflow_probability as tfp


class SAC(tf.Module):

    def __init__(self, policy, qfs, target_qfs,
                 policy_lr=3e-4, qf_lr=3e-4,
                 alpha_lr=3e-4, reward_scale=1.0,
                 discount=0.99, target_tau=5e-3,
                 constraint=-3e-2, target_delay=1):
        """An implementation of soft actor critic in static graph TensorFlow
        with automatic entropy tuning

        Args:

        policy: tf.keras.Model
            the policy neural network wrapped in a keras model
        q_functions: list of tf.keras.Model
            the q function neural network wrapped in a keras model
        target_q_functions: list of tf.keras.Model
            the target q function neural network wrapped in a keras model
        policy_lr: tf.float32
            the learning rate used to optimize the policy parameters
        qf_lr: tf.float32
            the learning rate used to optimize the q function parameters
        alpha_lr: tf.float32
            the learning rate used to optimize the lagrange multiplier
        reward_scale: tf.float32
            a scaling factor multiplied onto the environment reward
        discount: tf.float32
            the discount factor between 0 and 1
        target_tau: tf.float32
            the smoothing coefficient for the target q functions
        constraint: tf.float32
            the target value for the entropy of the policy
        target_delay: tf.int32
            the delay between updates to the target q functions
        """

        super(SAC, self).__init__()
        self.reward_scale = tf.constant(reward_scale)
        self.discount = tf.constant(discount)
        self.target_tau = tf.constant(target_tau)
        self.constraint = tf.constant(constraint)
        self.target_delay = tf.constant(target_delay)

        # create training machinery for the policy
        self.policy = policy
        self.policy_optim = \
            tf.keras.optimizers.Adam(lr=policy_lr, name="policy_optimizer")

        # create training machinery for alpha
        self.log_alpha = tf.Variable(-2.30258509299)
        self.alpha = tfp.util.DeferredTensor(self.log_alpha, tf.exp)
        self.alpha_optim = \
            tf.keras.optimizers.Adam(lr=alpha_lr, name='alpha_optimizer')

        # create training machinery for the q functions
        self.qfs = qfs
        self.target_qfs = target_qfs
        self.qfs_optim = tuple(tf.keras.optimizers.Adam(
            lr=qf_lr, name=f'qf{i}_optim') for i, q in enumerate(self.qfs))

        # reset the target networks at the beginning of training
        self.update_target(tf.constant(1.0))

    @tf.function
    def update_target(self, tau):
        """Perform a soft update to the parameters of the target critic using
        the value provided for the soft target tau

        Args:

        tau: tf.dtypes.float32
            a parameter to control the extent of the target update
        """

        # loop through every q function
        for qf, qf_t in zip(self.qfs, self.target_qfs):
            for source_weight, target_weight in zip(
                    qf.trainable_variables, qf_t.trainable_variables):

                # assign the target weights to a moving average
                target_weight.assign(tau * source_weight +
                                     (1.0 - tau) * target_weight)

    @tf.function
    def bellman_targets(self, reward, done, next_obs):
        """Calculate the bellman target values for the q function that
        will be regressed to using gradient descent

        Args:

        reward: tf.dtypes.float32
            a tensor shaped [batch_size, 1] containing a reward
        done: tf.dtypes.bool
            a tensor shaped [batch_size, 1] containing a done signal
        next_obs: tf.dtypes.float32
            a tensor shaped [batch_size, obs_size] containing observations
        """

        # sample new actions from the current policy
        distribution = self.policy(next_obs)
        act = distribution.sample()
        log_pis = distribution.log_prob(act)[:, tf.newaxis]

        # predict q values for the very next state
        next_q = [qf(tf.concat([next_obs, act], 1)) for qf in self.target_qfs]
        next_q = tf.reduce_min(next_q, axis=0) - self.alpha * log_pis

        # form the bellman target using a single bellman backup
        next_q = self.discount * (1.0 - tf.cast(done, next_q.dtype)) * next_q
        return tf.stop_gradient(next_q + self.reward_scale * reward)

    @tf.function
    def update_q(self, obs, act, reward, done, next_obs):
        """Perform a single gradient descent update on the q functions
        using a batch of data sampled from a replay buffer

        Args:

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

        # calculate the training labels for teh q functions
        bellman_targets = self.bellman_targets(reward, done, next_obs)
        for qf, qf_optim in zip(self.qfs, self.qfs_optim):
            with tf.GradientTape() as tape:

                # calculate the supervised loss for the q functions
                q_values = qf(tf.concat([obs, act], 1))
                q_loss = tf.keras.losses.logcosh(bellman_targets, q_values)
                q_loss = tf.reduce_mean(q_loss)

            # perform a step of gradient descent on the q function
            qf_optim.apply_gradients(zip(tape.gradient(
                q_loss, qf.trainable_variables), qf.trainable_variables))

    @tf.function
    def update_policy(self, obs):
        """Perform a single gradient descent update on the policy
        using a batch of data sampled from a replay buffer

        Args:

        obs: tf.dtypes.float32
            a tensor shaped [batch_size, obs_size] containing observations
        """

        with tf.GradientTape() as tape:
            # sample new actions using the current policy
            distribution = self.policy(obs)
            act = distribution.sample()
            log_pis = distribution.log_prob(act)[:, tf.newaxis]

            # compute the q values for the current policy
            q_log_targets = [qf(tf.concat([obs, act], 1)) for qf in self.qfs]
            q_log_targets = tf.reduce_min(q_log_targets, axis=0)

            # form the soft actor critic policy loss
            policy_loss = self.alpha * log_pis - q_log_targets
            policy_loss = tf.reduce_mean(policy_loss)

        # perform a step of gradient ascent on the policy
        policy_gradients = tape.gradient(
            policy_loss, self.policy.trainable_variables)
        self.policy_optim.apply_gradients(zip(
            policy_gradients, self.policy.trainable_variables))

    @tf.function
    def update_alpha(self, obs):
        """Perform a single gradient descent update on alpha
        using a batch of data sampled from a replay buffer

        Args:

        obs: tf.dtypes.float32
            a tensor shaped [batch_size, obs_size] containing observations
        """

        # sample new actions using the current policy
        distribution = self.policy(obs)
        act = distribution.sample()
        log_pis = distribution.log_prob(act)[:, tf.newaxis]

        # calculate the surrogate lagrangian loss for alpha
        with tf.GradientTape() as tape:
            alpha_loss = -self.alpha * tf.reduce_mean(
                log_pis + tf.reshape(self.constraint, [1, 1]))

        # update the lagrange multiplier using gradient descent
        self.alpha_optim.apply_gradients(zip(
            tape.gradient(alpha_loss, [self.log_alpha]), [self.log_alpha]))

    @tf.function
    def train(self, i, obs, act, reward, done, next_obs):
        """Perform a single gradient descent update on the agent
        using a batch of data sampled from a replay buffer

        Args:

        i: tf.dtypes.int64
            the scalar training iteration the agent is currently on
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

        # train the q functions policy and alpha every iteration
        self.update_q(obs, act, reward, done, next_obs)
        self.update_policy(obs)
        self.update_alpha(obs)

        # update the target networks after a delay
        if i % self.target_delay == 0:
            self.update_target(tau=self.target_tau)

    @tf.function
    def get_diagnostics(self, i, obs, act, reward, done, next_obs):
        """Gather diagnostic information from the learning algorithm,
        and return a dict containing tensors

        Args:

        i: tf.dtypes.int64
            the scalar training iteration the agent is currently on
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

        Returns:

        diagnostics: dict
            a dict containing tensors whose statistics will be summarized
        """

        # log the labels for the q function
        diagnostics = dict()
        diagnostics["sac/done"] = tf.cast(done, tf.float32)
        diagnostics["sac/bellman_targets"] = \
            self.bellman_targets(reward, done, next_obs)

        # log metrics for q function convergence
        for n, qf in enumerate(self.qfs):
            diagnostics[f"sac/q_values_{n}"] = qf(tf.concat([obs, act], 1))
            diagnostics[f"sac/q_loss_{n}"] = tf.keras.losses.logcosh(
                diagnostics["sac/bellman_targets"],
                diagnostics[f"sac/q_values_{n}"])

        # log metrics for tracking policy convergence
        d = self.policy(obs)
        act = d.sample()
        log_pis = d.log_prob(act)[:, tf.newaxis]
        diagnostics["sac/act"] = act
        diagnostics["sac/log_pis"] = log_pis
        diagnostics["sac/policy_loss"] = self.alpha * log_pis - \
            tf.reduce_min([qf(tf.concat([obs, act], 1))
                           for qf in self.qfs], axis=0)

        # log metrics for tracking the lagrange multiplier
        diagnostics["sac/alpha"] = self.alpha
        diagnostics["sac/alpha_loss"] = -self.alpha * (
            log_pis + tf.reshape(self.constraint, [1, 1]))
        return diagnostics
