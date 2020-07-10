import tensorflow as tf
from offpolicy.algorithms.algorithm import Algorithm


class TD3(Algorithm):

    def __init__(self,
                 policy,
                 target_policy,
                 q_functions,
                 target_q_functions,
                 policy_lr=tf.constant(1e-3),
                 q_lr=tf.constant(1e-3),
                 reward_scale=tf.constant(1.0),
                 discount=tf.constant(0.99),
                 tau=tf.constant(5e-3),
                 noise_std=tf.constant(0.2),
                 noise_range=tf.constant(0.5),
                 target_delay=tf.constant(2)):
        """An implementation of twin-delayed ddpg in static graph tensorflow
        using tf.keras models

        Args:

        policy: tf.keras.model
            the policy neural network wrapped in a keras model
        target_policy: tf.keras.model
            the target policy neural network wrapped in a keras model
        q_functions: list of tf.keras.model
            the q function neural network wrapped in a keras model
        target_q_functions: list of tf.keras.model
            the target q function neural network wrapped in a keras model
        """

        self.reward_scale = reward_scale
        self.discount = discount
        self.tau = tau
        self.noise_std = noise_std
        self.noise_range = noise_range
        self.target_delay = target_delay

        # create training machinery for the policy
        self.policy = policy
        self.target_policy = target_policy
        self.policy_optimizer = tf.optimizers.Adam(
            learning_rate=policy_lr, name="policy_optimizer")

        # create training machinery for the q functions
        self.q_functions = q_functions
        self.target_q_functions = target_q_functions
        self.q_optimizers = tuple(
            tf.optimizers.Adam(learning_rate=q_lr, name=f'q_{i}_optimizer')
            for i, q in enumerate(self.q_functions))

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

        for q, q_t in zip([self.policy] + self.q_functions,
                          [self.target_policy] + self.target_q_functions):
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

        act = self.target_policy.mean([next_obs])
        act = act + tf.clip_by_value(tf.random.normal(tf.shape(
            act)) * self.noise_std, -self.noise_range, self.noise_range)
        act = tf.clip_by_value(
            act, self.policy.low[tf.newaxis], self.policy.high[tf.newaxis])
        next_q = tuple(q([next_obs, act]) for q in self.target_q_functions)
        next_q = tf.reduce_min(next_q, axis=0)
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
                q_loss = tf.reduce_mean(tf.keras.losses.logcosh(
                    y_true=bellman_targets, y_pred=q([obs, act])))
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
            act = self.policy.mean([obs])
            policy_loss = -tf.reduce_mean(tf.reduce_min(
                tuple(q([obs, act]) for q in self.q_functions), axis=0))
        policy_gradients = tape.gradient(
            policy_loss, self.policy.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(
            policy_gradients, self.policy.trainable_variables))

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

        self.update_q(obs, act, reward, done, next_obs)
        self.update_policy(obs)
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

        new_act = self.policy.mean([obs])
        diagnostics = {
            "act": new_act,
            "done": tf.cast(done, tf.float32),
            "policy_loss": -tf.reduce_min(
                tuple(q([obs, new_act]) for q in self.q_functions), axis=0),
            "bellman_targets": self.bellman_targets(reward, done, next_obs)}
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

        saveables = {"policy": self.policy,
                     "policy_optim": self.policy_optimizer,
                     "target_policy": self.target_policy}
        for n, (q, optim) in enumerate(
                zip(self.q_functions, self.q_optimizers)):
            saveables[f"q_{n}"] = q
            saveables[f"q_optim_{n}"] = optim
            saveables[f"target_q_{n}"] = self.target_q_functions[n]
        return saveables
