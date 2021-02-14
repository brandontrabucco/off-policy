import tensorflow as tf
import tensorflow_probability as tfp
import gym
import tqdm
import os
from offpolicy.env import StaticGraphEnv
from offpolicy.sac import SAC
from offpolicy.replay_buffer import ReplayBuffer
from offpolicy.logger import Logger
from offpolicy.trainer import Trainer


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
        return tfp.distributions.TransformedDistribution(
            tfp.distributions.MultivariateNormalDiag(
                loc=mean, scale_diag=tf.math.softplus(logstd)), bijector)

    # build the neural network
    initializer = tf.keras.initializers.he_uniform()
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
    initializer = tf.keras.initializers.he_uniform()
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


if __name__ == "__main__":

    # training settings and hyper parameters
    logging_dir = "cheetah-1"
    env_name = "HalfCheetah-v2"

    # build the logger and reinforcement learning environment
    logger = Logger(logging_dir)
    eval_env = StaticGraphEnv(gym.make(env_name))
    training_env = StaticGraphEnv(gym.make(env_name))

    # build the replay buffer
    act_size = training_env.action_space.shape[0]
    obs_size = training_env.observation_space.shape[0]
    buffer = ReplayBuffer(1000000, obs_size, act_size)

    # create a deep neural network policy
    policy = make_policy(
        obs_size, act_size,
        training_env.action_space.low,
        training_env.action_space.high)

    # create multiple deep neural network q functions
    qf1 = make_qf(obs_size, act_size)
    qf2 = make_qf(obs_size, act_size)
    target_qf1 = make_qf(obs_size, act_size)
    target_qf2 = make_qf(obs_size, act_size)

    # build the reinforcement learning algorithm
    algorithm = SAC(
        policy, [qf1, qf2], [target_qf1, target_qf2],
        policy_lr=3e-4, qf_lr=3e-4, alpha_lr=3e-4,
        constraint=-float(act_size), reward_scale=1.0,
        discount=0.99, target_tau=5e-3, target_delay=1)

    # create an off policy training manager
    trainer = Trainer(
        training_env, eval_env, policy, buffer, algorithm,
        episodes_per_eval=10, warm_up_steps=5000, batch_size=256)

    # create a checkpoint manager for saving the algorithm
    algorithm_checkpoint = tf.train.CheckpointManager(
        tf.train.Checkpoint(algorithm=algorithm),
        os.path.join(logging_dir, 'algorithm'), max_to_keep=1)

    # create a checkpoint manager for saving the replay buffer
    buffer_checkpoint = tf.train.CheckpointManager(
        tf.train.Checkpoint(buffer=buffer),
        os.path.join(logging_dir, 'buffer'), max_to_keep=1)

    # load previous checkpoints from the disk
    algorithm_checkpoint.restore_or_initialize()
    buffer_checkpoint.restore_or_initialize()

    # train the algorithm for many iterations
    for iteration in tqdm.tqdm(list(range(5000000))):

        # perform a single training iteration
        trainer.train()

        if buffer.step % 10000 == 0:
            # save the algorithm and the replay buffer
            algorithm_checkpoint.save(checkpoint_number=buffer.step)
            buffer_checkpoint.save(checkpoint_number=buffer.step)

            # collect and log diagnostic information
            for key, value in trainer.get_diagnostics().items():
                logger.record(key, value, tf.cast(buffer.step, tf.int64))
