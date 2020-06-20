import tensorflow as tf
import gym
from sac.networks import FeedForward
from sac.networks import FeedForwardTanhGaussian
from sac.replay_buffer import ReplayBuffer
from sac.sac import SAC
from sac.logger import Logger
from sac.tensor_env import TensorEnv
from sac.trainer import Trainer


if __name__ == "__main__":

    logger = Logger("./half_cheetah/")
    training_env = TensorEnv(gym.make("HalfCheetah-v2"))
    eval_env = TensorEnv(gym.make("HalfCheetah-v2"))

    low = training_env.action_space.low
    high = training_env.action_space.high

    act_size = training_env.action_space.shape[0]
    obs_size = training_env.observation_space.shape[0]

    buffer = ReplayBuffer(1000000, obs_size, act_size)

    policy = FeedForwardTanhGaussian(256, act_size, low, high)

    q_functions = [FeedForward(256, 1), FeedForward(256, 1)]
    target_q_functions = [FeedForward(256, 1), FeedForward(256, 1)]
    sac = SAC(policy, q_functions, target_q_functions, logger)

    trainer = Trainer(training_env,
                      eval_env,
                      policy,
                      buffer,
                      sac,
                      logger)

    for i in range(1000000):
        trainer.train(tf.constant(i, tf.dtypes.int64))

        if i % 10000 == 0 and i > 0:
            policy.save("./half_cheetah/policy.h5")
            for j, q in enumerate(q_functions):
                q.save(f"./half_cheetah/q{j}.h5")
            for j, q in enumerate(target_q_functions):
                q.save(f"./half_cheetah/q{j}.h5")
