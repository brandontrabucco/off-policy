import tensorflow as tf
import gym
from sac.networks import FeedForward, FeedForwardTanhGaussian
from sac.replay_buffer import ReplayBuffer
from sac.sac import SAC
from sac.logger import Logger
from sac.tensor_env import TensorEnv
from sac.trainer import Trainer


if __name__ == "__main__":

    logger = Logger("./half_cheetah5/")
    training_env = TensorEnv(gym.make("HalfCheetah-v2"))
    eval_env = TensorEnv(gym.make("HalfCheetah-v2"))

    low = training_env.action_space.low
    high = training_env.action_space.high

    act_size = training_env.action_space.shape[0]
    obs_size = training_env.observation_space.shape[0]

    buffer = ReplayBuffer(1000000, obs_size, act_size)

    policy = FeedForwardTanhGaussian(256, act_size, low, high)
    q_functions = [FeedForward(256, 1),
                   FeedForward(256, 1)]
    target_q_functions = [FeedForward(256, 1),
                          FeedForward(256, 1)]

    algorithm = SAC(policy,
                    q_functions,
                    target_q_functions,
                    logger)

    trainer = Trainer(training_env,
                      eval_env,
                      policy,
                      buffer,
                      algorithm,
                      logger)

    for i in range(1000000):
        trainer.train(tf.constant(i, tf.dtypes.int64))
