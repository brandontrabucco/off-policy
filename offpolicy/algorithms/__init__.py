from offpolicy.algorithms.sac import SAC
from offpolicy.algorithms.td3 import TD3
from offpolicy.networks import FeedForward
from offpolicy.networks import TanhGaussian
from offpolicy.networks import Gaussian


def get_algorithm(alg,
                  low,
                  high,
                  obs_size,
                  act_size):
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

        policy = TanhGaussian(low, high, obs_size, 256, act_size)

        q_functions = [FeedForward(obs_size + act_size, 256, 1),
                       FeedForward(obs_size + act_size, 256, 1)]
        target_q_functions = [FeedForward(obs_size + act_size, 256, 1),
                              FeedForward(obs_size + act_size, 256, 1)]

        alg = SAC(policy,
                  q_functions,
                  target_q_functions)

    elif alg == "TD3":

        policy = Gaussian(low, high, obs_size, 256, act_size)
        target_policy = Gaussian(low, high, obs_size, 256, act_size)

        q_functions = [FeedForward(obs_size + act_size, 256, 1),
                       FeedForward(obs_size + act_size, 256, 1)]
        target_q_functions = [FeedForward(obs_size + act_size, 256, 1),
                              FeedForward(obs_size + act_size, 256, 1)]

        alg = TD3(policy,
                  target_policy,
                  q_functions,
                  target_q_functions)

    return alg
