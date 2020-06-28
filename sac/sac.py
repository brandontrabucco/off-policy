import tensorflow as tf
import tensorflow_probability as tfp


class SAC(tf.Module):

    def __init__(self,
                 policy,
                 q_functions,
                 target_q_functions,
                 policy_lr=tf.constant(3e-4),
                 q_lr=tf.constant(3e-4),
                 alpha_lr=tf.constant(3e-4),
                 reward_scale=tf.constant(1.0),
                 discount=tf.constant(0.99),
                 tau=tf.constant(5e-3),
                 target_delay=tf.constant(1)):
        """An implementation of soft actor critic in static graph tensorflow
        with automatic entropy tuning

        Args:

        policy: tf.keras.model
            the policy neural network wrapped in a keras model
        q_functions: list of tf.keras.model
            the q function neural network wrapped in a keras model
        target_q_functions: list of tf.keras.model
            the target q function neural network wrapped in a keras model
        """

        super(SAC, self).__init__()
        self.reward_scale = reward_scale
        self.discount = discount
        self.tau = tau
        self.target_delay = target_delay

        # create training machinery for the policy
        self.policy = policy
        self.policy_optimizer = tf.optimizers.Adam(
            learning_rate=policy_lr, name="policy_optimizer")

        # create training machinery for the q functions
        self.q_functions = q_functions
        self.target_q_functions = target_q_functions
        self.q_optimizers = tuple(
            tf.optimizers.Adam(learning_rate=q_lr, name=f'q_{i}_optimizer')
            for i, q in enumerate(self.q_functions))

        # create training machinery for alpha
        self.log_alpha = tf.Variable(0.0)
        self.alpha = tfp.util.DeferredTensor(self.log_alpha, tf.exp)
        self.alpha_optimizer = tf.optimizers.Adam(
            learning_rate=alpha_lr, name='alpha_optimizer')

    @tf.function
    def update_target(self, tau):
        """Perform a soft update to the parameters of the target critic using
        the value provided for the soft target tau

        Args:

        tau: tf.dtypes.float32
            a parameter to control the extent of the target update
        """

        for q, q_t in zip(self.q_functions, self.target_q_functions):
            for source_weight, target_weight in zip(
                    q.trainable_variables, q_t.trainable_variables):
                target_weight.assign(
                    tau * source_weight + (1.0 - tau) * target_weight)

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

        act, log_pis = self.policy.sample([next_obs], log_probs=True)
        next_q = tuple(q([next_obs, act]) for q in self.target_q_functions)
        next_q = tf.reduce_min(next_q, axis=0) - self.alpha * log_pis
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

        bellman_targets = self.bellman_targets(reward, done, next_obs)
        for n, (q, optim) in enumerate(
                zip(self.q_functions, self.q_optimizers)):
            with tf.GradientTape() as tape:
                q_values = q([obs, act])
                q_loss = tf.keras.losses.logcosh(
                    y_true=bellman_targets, y_pred=q_values)
                q_loss = tf.reduce_mean(q_loss)
            optim.apply_gradients(zip(tape.gradient(
                q_loss, q.trainable_variables), q.trainable_variables))

    @tf.function
    def update_policy(self, obs):
        """Perform a single gradient descent update on the policy
        using a batch of data sampled from a replay buffer

        Args:

        obs: tf.dtypes.float32
            a tensor shaped [batch_size, obs_size] containing observations
        """

        with tf.GradientTape() as tape:
            act, log_pis = self.policy.sample([obs], log_probs=True)
            q_log_targets = tuple(q([obs, act]) for q in self.q_functions)
            q_log_targets = tf.reduce_min(q_log_targets, axis=0)
            policy_loss = self.alpha * log_pis - q_log_targets
            policy_loss = tf.reduce_mean(policy_loss)
        policy_gradients = tape.gradient(
            policy_loss, self.policy.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(
            policy_gradients, self.policy.trainable_variables))

    @tf.function
    def update_alpha(self, obs):
        """Perform a single gradient descent update on alpha
        using a batch of data sampled from a replay buffer

        Args:

        obs: tf.dtypes.float32
            a tensor shaped [batch_size, obs_size] containing observations
        """

        act, log_pis = self.policy.sample([obs], log_probs=True)
        with tf.GradientTape() as tape:
            alpha_loss = -self.log_alpha * tf.stop_gradient(
                log_pis - tf.cast(tf.shape(act)[-1], act.dtype))
            alpha_loss = tf.reduce_mean(alpha_loss)
        self.alpha_optimizer.apply_gradients(zip(
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

        if tf.equal(i, 0):
            self.update_target(tf.constant(1.0))
        self.update_q(obs, act, reward, done, next_obs)
        self.update_policy(obs)
        self.update_alpha(obs)
        if i % self.target_delay == 0:
            self.update_target(tau=self.tau)

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

        new_act, log_pis = self.policy.sample([obs], log_probs=True)
        diagnostics = {
            "act": new_act, "log_pis": log_pis,
            "policy_loss": self.alpha * log_pis - tf.reduce_min(
                tuple(q([obs, new_act]) for q in self.q_functions), axis=0),
            "bellman_targets": self.bellman_targets(reward, done, next_obs),
            "alpha_loss": -self.log_alpha * tf.stop_gradient(
                log_pis - tf.cast(tf.shape(act)[-1], act.dtype))}
        for n, (q, optim) in enumerate(
                zip(self.q_functions, self.q_optimizers)):
            diagnostics[f"q_values_{n}"] = q([obs, act])
            diagnostics[f"q_loss_{n}"] = tf.keras.losses.logcosh(
                diagnostics["bellman_targets"], diagnostics[f"q_values_{n}"])
        return diagnostics

    def get_saveables(self):
        """Collects and returns stateful objects that are serializeable
        using the tensorflow checkpoint format

        Returns:

        saveables: dict
            a dict containing stateful objects compatible with checkpoints
        """

        saveables = {"log_alpha": self.log_alpha,
                     "log_alpha_optim": self.alpha_optimizer,
                     "policy": self.policy,
                     "policy_optim": self.policy_optimizer}
        for n, (q, optim) in enumerate(
                zip(self.q_functions, self.q_optimizers)):
            saveables[f"q_{n}"] = q
            saveables[f"q_optim_{n}"] = optim
            saveables[f"target_q_{n}"] = self.target_q_functions[n]
        return saveables
