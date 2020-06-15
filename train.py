from sac.networks import FeedForward, FeedForwardTanhGaussian
from sac.replay_buffer import ReplayBuffer
from sac.sac import SAC
from sac.logger import Logger
from sac.tensor_env import TensorEnv
from sac.trainer import Trainer
from sac.envs import PointMass


if __name__ == "__main__":

    logger = Logger("./point_mass/")
    env = TensorEnv(PointMass())

    f = env.wrapped_env.render(mode="rgb_array")
    print(f.shape)
    exit()

    act_size = env.action_space.shape[0]
    obs_size = env.observation_space.shape[0]
    ctx_size = env.context_space.shape[0]

    policy = FeedForwardTanhGaussian(
        256, act_size, env.action_space.low, env.action_space.high)
    q_functions = [FeedForward(256, 1), FeedForward(256, 1)]
    target_q_functions = [FeedForward(256, 1), FeedForward(256, 1)]

    algorithm = SAC(policy, q_functions, target_q_functions, logger=logger)
    buffer = ReplayBuffer(100000, obs_size, ctx_size, act_size)
    trainer = Trainer(env, policy, buffer, algorithm, logger)

    trainer.train(1000000)
