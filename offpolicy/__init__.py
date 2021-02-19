import tensorflow as tf
import tensorflow_probability as tfp
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


def make_policy(obs_size, act_size, low, high, hidden_size=256):
    """Creates a deep neural network policy that parameterizes a
    probability distribution over actions

    Args:

    obs_size: int
        the number of units in the input layer of the network
    act_size: int
        the number of units in the output layer of the network
    low: tf.Tensor
        the lower bound on actions the network can generate
    high: tf.Tensor
        the upper bound on actions the network can generate
    hidden_size: int
        the number of units in hidden layers in the network

    Returns:

    model: tf.keras.Sequential
        a  model that maps observations to a distribution over actions
    """

    # create a tanh bijector that has a defined mean
    hacked_tanh_bijector = tfp.bijectors.Tanh()
    hacked_tanh_bijector._is_constant_jacobian = True
    bijector = tfp.bijectors.Chain([
        tfp.bijectors.Shift(shift=(high + low)[tf.newaxis] / 2.0),
        tfp.bijectors.Scale(scale=(high - low)[tf.newaxis] / 2.0),
        hacked_tanh_bijector])

    # define a function that creates a probability distribution
    def create_d(x):
        mean, logstd = tf.split(x, 2, axis=-1)
        logstd = tf.clip_by_value(logstd, -6.0, 6.0)
        return tfp.distributions.TransformedDistribution(
            tfp.distributions.MultivariateNormalDiag(
                loc=mean, scale_diag=tf.math.softplus(logstd)), bijector)

    # build the neural network
    initializer = tf.keras.initializers.GlorotUniform()
    return tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_size,
                              input_shape=(obs_size,),
                              kernel_initializer=initializer),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(hidden_size,
                              kernel_initializer=initializer),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(act_size * 2,
                              kernel_initializer=initializer),
        tfp.layers.DistributionLambda(create_d)])


def make_qf(obs_size, act_size, hidden_size=256):
    """Creates a deep neural network policy that parameterizes a
    probability distribution over actions

    Args:

    obs_size: int
        the number of units in the observation vector
    act_size: int
        the number of units in the action vector
    hidden_size: int
        the number of units in hidden layers in the network

    Returns:

    model: tf.keras.Sequential
        a  model that maps observations and actions to q values
    """

    # build the neural network
    initializer = tf.keras.initializers.GlorotUniform()
    return tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_size,
                              input_shape=(obs_size + act_size,),
                              kernel_initializer=initializer),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(hidden_size,
                              kernel_initializer=initializer),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(1,
                              kernel_initializer=initializer)])


def soft_actor_critic(config):
    """Train a reinforcement learning agent using an off-policy
    reinforcement learning algorithm

    Args:

    config: dict
        a dictionary of hyper parameters that are passed to the
        reinforcement learning algorithm

        Includes:

        logging_dir: str
            the directory where checkpoints are periodically saved
        eval_env: str
            the string passed to gym.make to create the eval environment
        training_env: str
            the string passed to gym.make to create the train environment
        buffer_capacity: int
            the maximum number of transitions in the replay buffer
        hidden_size: int
            the number of units in the policy hidden layers

        policy_lr: float
            the learning rate used to optimize the policy parameters
        qf_lr: float
            the learning rate used to optimize the q function parameters
        alpha_lr: float
            the learning rate used to optimize the lagrange multiplier
        reward_scale: float
            a scaling factor multiplied onto the environment reward
        discount: float
            the discount factor between 0 and 1
        target_tau: float
            the smoothing coefficient for the target q functions
        constraint: float
            the target value for the entropy of the policy
        target_delay: int
            the delay between updates to the target q functions

        episodes_per_eval: int
            the number of episodes to collect per evaluation
        warm_up_steps: int
            the number of steps to use a random uniform exploration policy
        batch_size: int
            the number of samples per training batch
        normalizer_scale: float
            multiplied onto std of observations during normalization
        normalizer_range: float
            range of normalized observations to clip values to stay within

        training_iterations: int
            the number of total training iterations in this session
        eval_interval: int
            the number of training updates per evaluation
    """

    from offpolicy.env import StaticGraphEnv
    from offpolicy.sac import SAC
    from offpolicy.replay_buffer import ReplayBuffer
    from offpolicy.logger import Logger
    from offpolicy.trainer import Trainer
    import gym
    import os

    # hyper parameters for the experimental trial
    logging_dir = config["logging_dir"]
    eval_env = config["eval_env"]
    training_env = config["training_env"]
    buffer_capacity = config["buffer_capacity"]
    hidden_size = config["hidden_size"]

    # hyper parameters for the learning algorithm
    policy_lr = config["policy_lr"]
    qf_lr = config["qf_lr"]
    alpha_lr = config["alpha_lr"]
    constraint = config["constraint"]
    reward_scale = config["reward_scale"]
    discount = config["discount"]
    target_tau = config["target_tau"]
    target_delay = config["target_delay"]

    # hyper parameters for the training manager
    episodes_per_eval = config["episodes_per_eval"]
    warm_up_steps = config["warm_up_steps"]
    batch_size = config["batch_size"]
    normalizer_scale = config["normalizer_scale"]
    normalizer_range = config["normalizer_range"]

    # hyper parameters for the training loop
    training_iterations = config["training_iterations"]
    eval_interval = config["eval_interval"]

    # build the logger and reinforcement learning environment
    logger = Logger(logging_dir)
    eval_env = StaticGraphEnv(gym.make(eval_env))
    training_env = StaticGraphEnv(gym.make(training_env))

    # build the replay buffer
    act_size = training_env.action_space.shape[0]
    obs_size = training_env.observation_space.shape[0]
    buffer = ReplayBuffer(buffer_capacity, obs_size, act_size)

    # create a deep neural network policy
    policy = make_policy(
        obs_size, act_size, training_env.action_space.low,
        training_env.action_space.high, hidden_size=hidden_size)

    # create multiple deep neural network q functions
    qf1 = make_qf(obs_size, act_size, hidden_size=hidden_size)
    qf2 = make_qf(obs_size, act_size, hidden_size=hidden_size)
    target_qf1 = make_qf(obs_size, act_size, hidden_size=hidden_size)
    target_qf2 = make_qf(obs_size, act_size, hidden_size=hidden_size)

    # build the reinforcement learning algorithm
    algorithm = SAC(
        policy, [qf1, qf2], [target_qf1, target_qf2],
        policy_lr=policy_lr, qf_lr=qf_lr, alpha_lr=alpha_lr,
        constraint=-float(act_size) if constraint is None else constraint,
        reward_scale=reward_scale, discount=discount,
        target_tau=target_tau, target_delay=target_delay)

    # create an off policy training manager
    trainer = Trainer(
        training_env, eval_env, policy, buffer, algorithm,
        episodes_per_eval=episodes_per_eval, warm_up_steps=warm_up_steps,
        batch_size=batch_size, normalizer_scale=normalizer_scale,
        normalizer_range=normalizer_range)

    # create a checkpoint manager for saving the replay buffer
    buffer_checkpoint = tf.train.CheckpointManager(
        tf.train.Checkpoint(buffer=buffer),
        os.path.join(logging_dir, 'buffer'), max_to_keep=1)

    # create a checkpoint manager for saving the algorithm
    algorithm_checkpoint = tf.train.CheckpointManager(
        tf.train.Checkpoint(algorithm=algorithm),
        os.path.join(logging_dir, 'algorithm'), max_to_keep=1)

    # load previous checkpoints from disk
    buffer_checkpoint.restore_or_initialize()
    algorithm_checkpoint.restore_or_initialize()

    # train the algorithm for many iterations
    for iteration in range(training_iterations):
        # perform a single training iteration
        trainer.train()

        if buffer.step % eval_interval == 0:
            # save the algorithm and the replay buffer
            buffer_checkpoint.save(checkpoint_number=buffer.step)
            algorithm_checkpoint.save(checkpoint_number=buffer.step)

            # collect and log diagnostic information
            for key, value in trainer.get_diagnostics().items():
                logger.record(key, value, tf.cast(buffer.step, tf.int64))
