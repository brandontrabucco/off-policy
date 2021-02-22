from offpolicy import soft_actor_critic
from ray import tune
import ray
import os
import click


@click.command()
@click.option('--logging-dir', type=str, default='hopper')
@click.option('--cpus', type=int, default=4)
@click.option('--gpus', type=int, default=1)
@click.option('--num-parallel', type=int, default=1)
@click.option('--num-samples', type=int, default=1)
def tune_hyper_parameters(logging_dir, cpus, gpus, num_parallel, num_samples):
    """Train a reinforcement learning agent using an off-policy
    reinforcement learning algorithm

    Args:

    logging_dir: str
        the directory where checkpoints are periodically saved
    cpus: int
        the total number of cpu cores allocated to this session
    gpus: int
        the total number of gpu nodes allocated to this session
    num_parallel: int
        the number of experimental trials to launch at the same time
    num_samples: int
        the number of trials per hyper parameter setting to launch
    """

    # hyper parameters for the experimental trial
    config = dict(logging_dir="data",
                  eval_env="Hopper-v2",
                  training_env="Hopper-v2",
                  buffer_capacity=1000000,
                  hidden_size=256,
                  policy_lr=3e-4,
                  qf_lr=3e-4,
                  alpha_lr=3e-4,
                  constraint=None,
                  reward_scale=1.0,
                  discount=0.99,
                  target_tau=5e-3,
                  target_delay=1,
                  episodes_per_eval=10,
                  warm_up_steps=5000,
                  batch_size=256,
                  variance_scaling=1.0,
                  clip_range=None,
                  training_iterations=5000000,
                  eval_interval=10000)

    # start the ray interface
    ray.init(num_cpus=cpus,
             num_gpus=gpus,
             include_dashboard=False,
             _temp_dir=os.path.expanduser('~/tmp'))

    # expand relative paths padded to ray
    logging_dir = os.path.abspath(logging_dir)

    # launch a hyper parameter tuning session
    tune.run(soft_actor_critic, config=config, num_samples=num_samples,
             name=os.path.basename(logging_dir),
             trial_dirname_creator=lambda t: f"trial-{t.experiment_tag}",
             local_dir=os.path.dirname(logging_dir), resources_per_trial={
                'cpu': cpus // num_parallel,
                'gpu': gpus / num_parallel - 1e-3})


if __name__ == "__main__":
    tune_hyper_parameters()
