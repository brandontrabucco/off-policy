import tensorflow as tf

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


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

    from offpolicy.replay_buffer import ReplayBuffer
    from offpolicy.algorithms import get_algorithm
    from offpolicy.logger import Logger
    from offpolicy.env import Env
    from offpolicy.trainer import Trainer
    import os

    n = 1000000
    logger = Logger(logging_dir)
    training_env = Env(training_env)
    eval_env = Env(eval_env)

    act_size = training_env.action_space.shape[0]
    obs_size = training_env.observation_space.shape[0]

    alg = get_algorithm(
        alg,
        training_env.action_space.low,
        training_env.action_space.high,
        obs_size,
        act_size)

    b = ReplayBuffer(1000000, obs_size, act_size)

    trainer = Trainer(
        training_env,
        eval_env,
        alg.policy,
        b,
        alg)

    ckpt = tf.train.Checkpoint(**trainer.get_saveables())
    path = tf.train.latest_checkpoint(logging_dir)
    if path is not None:
        ckpt.restore(path)

    while b.step < n:

        trainer.train()

        if b.step % 5000 == 0:
            for key, value in trainer.get_diagnostics().items():
                logger.record(key, value, tf.cast(b.step, tf.int64))

        if b.step % 10000 == 0 and b.step > 0:
            ckpt.save(os.path.join(logging_dir, 'alg.ckpt'))
