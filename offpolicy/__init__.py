import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


def train(logging_dir,
          training_env,
          eval_env,
          alg,
          **kwargs):
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
        act_size,
        **kwargs)

    b = ReplayBuffer(
        kwargs.get('buffer_size', 1000000), obs_size, act_size)

    trainer = Trainer(
        training_env,
        eval_env,
        alg.policy,
        b,
        alg,
        warm_up_steps=kwargs.get('warm_up_steps', 5000),
        batch_size=kwargs.get('batch_size', 256))

    ckpt = tf.train.Checkpoint(**trainer.get_saveables())
    manager = tf.train.CheckpointManager(
        ckpt, directory=logging_dir, max_to_keep=5)
    manager.restore_or_initialize()

    while b.step < kwargs.get('iterations', 1000000):
        trainer.train()
        if b.step % kwargs.get('log_interval', 10000) == 0:
            manager.save()
            for key, value in trainer.get_diagnostics().items():
                logger.record(key, value, tf.cast(b.step, tf.int64))
