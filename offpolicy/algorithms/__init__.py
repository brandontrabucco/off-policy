from offpolicy.algorithms.sac import SAC
from offpolicy.algorithms.td3 import TD3
from offpolicy.networks import FeedForward
from offpolicy.networks import TanhGaussian
from offpolicy.networks import Gaussian
import tensorflow as tf


def get_algorithm(alg,
                  low,
                  high,
                  obs_size,
                  act_size,
                  **kwargs):
    """Creates an algorithm for training a policy with reinforcement
    learning and returns an instance of Algorithm

    Args:

    alg: str
        the name of the algorithm to build, supports SAC and TD3
    low: tf.dtypes.float32
        the lower bound of the action space for the policy
    high: tf.dtypes.float32
        the upper bound of the action space for the policy
    obs_size: tf.dtypes.int32
        the number of channels in the flattened observations of the agent
    act_size: tf.dtypes.int32
        the number of channels in the flattened actions of the agent

    Returns:

    alg: Algorithm
        an instance of Algorithm, which trains a policy with rl
    """

    if alg == "SAC":

        policy = TanhGaussian(
            low, high, obs_size, 256, act_size)

        q_functions = [
            FeedForward(obs_size + act_size, 256, 1),
            FeedForward(obs_size + act_size, 256, 1)]

        target_q_functions = [
            FeedForward(obs_size + act_size, 256, 1),
            FeedForward(obs_size + act_size, 256, 1)]

        alg = SAC(
            policy,
            q_functions,
            target_q_functions,
            policy_lr=tf.constant(kwargs.get('policy_lr', 3e-4)),
            q_lr=tf.constant(kwargs.get('q_lr', 3e-4)),
            alpha_lr=tf.constant(kwargs.get('alpha_lr', 3e-4)),
            reward_scale=tf.constant(kwargs.get('reward_scale', 1.0)),
            discount=tf.constant(kwargs.get('discount', 0.99)),
            tau=tf.constant(kwargs.get('tau', 5e-3)),
            target_delay=tf.constant(kwargs.get('target_delay', 1)))

    elif alg == "TD3":

        policy = Gaussian(
            low, high, obs_size, 400, act_size,
            expl_noise=tf.constant(kwargs.get('expl_noise', 0.1)))

        target_policy = Gaussian(
            low, high, obs_size, 400, act_size,
            expl_noise=tf.constant(kwargs.get('expl_noise', 0.1)))

        q_functions = [
            FeedForward(obs_size + act_size, 400, 1),
            FeedForward(obs_size + act_size, 400, 1)]

        target_q_functions = [
            FeedForward(obs_size + act_size, 400, 1),
            FeedForward(obs_size + act_size, 400, 1)]

        alg = TD3(
            policy,
            target_policy,
            q_functions,
            target_q_functions,
            policy_lr=tf.constant(kwargs.get('policy_lr', 1e-3)),
            q_lr=tf.constant(kwargs.get('q_lr', 1e-3)),
            reward_scale=tf.constant(kwargs.get('reward_scale', 1.0)),
            discount=tf.constant(kwargs.get('discount', 0.99)),
            tau=tf.constant(kwargs.get('tau', 5e-3)),
            noise_std=tf.constant(kwargs.get('noise_std', 0.2)),
            noise_range=tf.constant(kwargs.get('noise_range', 0.5)),
            target_delay=tf.constant(kwargs.get('target_delay', 2)))

    return alg
