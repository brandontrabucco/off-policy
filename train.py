from offpolicy.replay_buffer import ReplayBuffer
from offpolicy.algorithms import get_algorithm
from offpolicy.logger import Logger
from offpolicy.tensor_env import TensorEnv
from offpolicy.trainer import Trainer
import tensorflow as tf
import os
import gym


def train(logging_dir,
          training_env,
          eval_env,
          alg):
    """Train a policy using an off policy reinforcement learning
    algorithm such as SAC or TD3

    Args:

    logging_dir: str
        the disk path to where metrics and weights will be saved
    training_env: gym.Env
        an environment on which a policy shall be trained
    eval_env: gym.Env
        an environment on which a policy shall be evaluated
    alg: str
        a string identifier that indicates which algorithm to use
    """

    logger = Logger(logging_dir)
    training_env = TensorEnv(training_env)
    eval_env = TensorEnv(eval_env)

    act_size = training_env.action_space.shape[0]
    obs_size = training_env.observation_space.shape[0]

    alg = get_algorithm(alg,
                        training_env.action_space.low,
                        training_env.action_space.high,
                        obs_size,
                        act_size)

    trainer = Trainer(
        training_env,
        eval_env,
        alg.policy,
        ReplayBuffer(1000000, obs_size, act_size),
        alg)

    ckpt = tf.train.Checkpoint(**trainer.get_saveables())

    latest_ckpt = tf.train.latest_checkpoint(logging_dir)
    if latest_ckpt is not None:
        ckpt.restore(latest_ckpt)

    for i in range(1000000):
        trainer.train()
        if i % 5000 == 0:
            for key, value in trainer.get_diagnostics().items():
                logger.record(key, value, tf.cast(i, tf.dtypes.int64))
        if i % 10000 == 0 and i > 0:
            ckpt.save(os.path.join(logging_dir, 'alg.ckpt'))


if __name__ == "__main__":

    train('./half_cheetah_td32',
          gym.make('HalfCheetah-v2'),
          gym.make('HalfCheetah-v2'),
          'TD3')
