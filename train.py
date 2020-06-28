from sac.networks import FeedForward
from sac.networks import TanhGaussian
from sac.replay_buffer import ReplayBuffer
from sac.sac import SAC
from sac.logger import Logger
from sac.tensor_env import TensorEnv
from sac.trainer import Trainer
import tensorflow as tf
import os
import gym


if __name__ == "__main__":

    logging_dir = './half_cheetah7'

    logger = Logger(logging_dir)
    training_env = TensorEnv(gym.make("HalfCheetah-v2"))
    eval_env = TensorEnv(gym.make("HalfCheetah-v2"))

    low = training_env.action_space.low
    high = training_env.action_space.high

    act_size = training_env.action_space.shape[0]
    obs_size = training_env.observation_space.shape[0]

    buffer = ReplayBuffer(1000000, obs_size, act_size)

    policy = TanhGaussian(low, high, obs_size, 256, act_size)

    q_functions = [
        FeedForward(obs_size + act_size, 256, 1),
        FeedForward(obs_size + act_size, 256, 1)]

    target_q_functions = [
        FeedForward(obs_size + act_size, 256, 1),
        FeedForward(obs_size + act_size, 256, 1)]

    sac = SAC(policy,
              q_functions,
              target_q_functions)

    trainer = Trainer(training_env,
                      eval_env,
                      sac.policy,
                      buffer,
                      sac)

    ckpt = tf.train.Checkpoint(**trainer.get_saveables())

    latest_ckpt = tf.train.latest_checkpoint(logging_dir)
    if latest_ckpt is not None:
        ckpt.restore(latest_ckpt)
        print(f'loading checkpoint from {latest_ckpt}')

    for i in range(1000000):

        trainer.train()

        if i % 5000 == 0:
            for key, value in trainer.get_diagnostics().items():
                logger.record(key, value, tf.cast(i, tf.dtypes.int64))

        if i % 10000 == 0 and i > 0:
            ckpt.save(os.path.join(logging_dir, 'ckpt'))
