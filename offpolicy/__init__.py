import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


DEFAULT_KWARGS = dict(
    logdir='ant',
    env='Ant-v2',
    policy_lr=3e-4,
    q_lr=3e-4,
    alpha_lr=3e-4,
    reward_scale=1.0,
    discount=0.99,
    tau=5e-3,
    target_entropy=-3e-2,
    target_delay=1,
    buffer_size=1000000,
    episodes_per_eval=10,
    warm_up_steps=10000,
    batch_size=256,
    max_to_keep=5,
    checkpoint_interval=10000,
    iterations=1000000,
    log_interval=10000)


def train(logging_dir,
          training_env,
          eval_env,
          **kwargs):
    """Train a policy using an off policy reinforcement learning
    algorithm such as SAC

    Args:

    logging_dir: str
        the disk path to where metrics and weights will be saved
    training_env: gym.Env
        an environment on which a policy shall be trained
    eval_env: gym.Env
        an environment on which a policy shall be evaluated
    """

    from offpolicy.replay_buffer import ReplayBuffer
    from offpolicy.nets import FeedForward
    from offpolicy.nets import TanhGaussian
    from offpolicy.sac import SAC
    from offpolicy.logger import Logger
    from offpolicy.env import StaticGraphEnv
    from offpolicy.trainer import Trainer

    logger = Logger(logging_dir)
    training_env = StaticGraphEnv(training_env)
    eval_env = StaticGraphEnv(eval_env)

    act_size = training_env.action_space.shape[0]
    obs_size = training_env.observation_space.shape[0]

    policy = TanhGaussian(
        training_env.action_space.low,
        training_env.action_space.high,
        obs_size, 256, act_size)

    q_functions = [
        FeedForward(obs_size + act_size, 256, 1),
        FeedForward(obs_size + act_size, 256, 1)]

    target_q_functions = [
        FeedForward(obs_size + act_size, 256, 1),
        FeedForward(obs_size + act_size, 256, 1)]

    alg = SAC(
        policy, q_functions, target_q_functions,
        policy_lr=tf.constant(kwargs.get('policy_lr', 3e-4)),
        q_lr=tf.constant(kwargs.get('q_lr', 3e-4)),
        alpha_lr=tf.constant(kwargs.get('alpha_lr', 3e-4)),
        reward_scale=tf.constant(kwargs.get('reward_scale', 1.0)),
        discount=tf.constant(kwargs.get('discount', 0.99)),
        tau=tf.constant(kwargs.get('tau', 5e-3)),
        target_entropy=tf.constant(kwargs.get('target_entropy', -3e-2)),
        target_delay=tf.constant(kwargs.get('target_delay', 1)))

    b = ReplayBuffer(
        kwargs.get('buffer_size', 1000000), obs_size, act_size)

    trainer = Trainer(
        training_env, eval_env, policy, b, alg,
        episodes_per_eval=kwargs.get('episodes_per_eval', 10),
        warm_up_steps=kwargs.get('warm_up_steps', 10000),
        batch_size=kwargs.get('batch_size', 256))

    manager = tf.train.CheckpointManager(
        tf.train.Checkpoint(**trainer.get_saveables()),
        directory=logging_dir,
        step_counter=b.step,
        max_to_keep=kwargs.get('max_to_keep', 5),
        checkpoint_interval=kwargs.get('checkpoint_interval', 1000))

    manager.restore_or_initialize()
    while b.step < kwargs.get('iterations', 1000000):
        trainer.train()
        manager.save(checkpoint_number=b.step)
        if b.step % kwargs.get('log_interval', 10000) == 0:
            for key, value in trainer.get_diagnostics().items():
                logger.record(key, value, tf.cast(b.step, tf.int64))
