# Off-Policy Reinforcement Learning

A high-performing minimalist implementation of SAC meant for rapid prototyping. Have fun! -Brandon

## Setup

I've set up this framework as a pip package, which you can install like this.

```bash
git clone https://github.com/brandontrabucco/off-policy.git
pip install -e off-policy
```

## Usage

The entry point for training is in the aptly named `train.py`. You can train a policy for an environment that implements `gym.Env`. Training supports a number of hyperparameters specified via the command line. For example, lets trian a policy for the gym `Ant-v2` environment.

```bash
python train.py --logdir ant \
                --env Ant-v2 \
                --policy_lr 3e-4 \
                --q_lr 3e-4 \
                --alpha_lr 3e-4 \
                --reward_scale 1.0 \
                --discount 0.99 \
                --tau 5e-3 \
                --target_entropy -8.0 \
                --target_delay 1 \
                --buffer_size 1000000 \
                --episodes_per_eval 10 \
                --warm_up_steps 10000 \
                --batch_size 256 \
                --max_to_keep 5 \
                --checkpoint_interval 10000 \
                --iterations 1000000 \
                --log_interval 10000
```

### Checkpoints

Checkpoints and diagnostic information will be saved in a new `ant` folder given by the `--logdir` command line argument. This folder will contain a file with a name like `events.out.tfevents.1594452506.desktop.9612.13.v2`, and several checkpoints with a name like `ckpt-100001.data-00000-of-00004`. When training initializes, an existing checkpoint will be loaded from the `--logdir` folder, which includes the replay buffer, optimizers, and neural network weights.

### Visualization

Diagnostic information is recorded using tensorboard, and can be visualized using tensorboard. 

```bash
tensorboard --logdir ant
```

### Implementing Your Algorithms

The heavy lifting in this framework is done in `sac.py`, which manages several neural networks, q functions and policies. Most common off-policy reinforcement learning algorithms can be implemented by modifying the `SAC` class. Alternatively, if you require more sophisticated behavior, such as simulating rollouts from the environment using a fitted dynamics model, you would modify `trainer.py`.

The final step is then to modify the `train()` function in the `__init__.py` to build your algorithm. By default, the framework supports and will save arbitrary command line arguments, so there is no need to modify the entry point in `train.py` with custom behavior. A `kwargs.json` file with your hyperparameters will be saved.

### Efficiency 

The framework is currently optimized for GPU performance, and is typically faster when using a GPU, even for state-based experiments. For example, when training a policy for `Ant-v2`, reaching 1M gradient descent steps typically requires 3 hours when using an NVIDIA Titan X Pascal.
