from offpolicy import train, DEFAULT_KWARGS
import numpy as np
import gym
import sys
import os
import json
import copy


def parse_args(args):
    """Parse arbitrary command line arguments into a dictionary of
    parameters for specifying hyper-parameters
    https://stackoverflow.com/a/34097314

    Args:

    args: list of str
        a list of strings that specify command line args and values

    Returns:

    dict: dict of str: value
        a dictionary mapping names to their corresponding values
    """

    ret_args = dict()
    for index, k in enumerate(args):

        # check if there are additional args left
        if index < len(args) - 1:
            a, b = k, args[index+1]
        else:
            a, b = k, None

        new_key = None

        # double hyphen, equals
        if a.startswith('--') and '=' in a:
            new_key, val = a.split('=')

        # double hyphen, no arg
        elif a.startswith('--') and (not b or b.startswith('--')):
            val = True

        # double hypen, arg
        elif a.startswith('--') and b and not b.startswith('--'):
            val = b

        # there are no args left
        else:
            continue

        # parse int
        if isinstance(val, str):
            try:
                val = int(val)
            except ValueError:
                pass

        # parse float
        if isinstance(val, str):
            try:
                val = float(val)
            except ValueError:
                pass

        # santize the key
        ret_args[(new_key or a).strip(' -')] = val

    return ret_args


if __name__ == "__main__":

    kwargs = copy.copy(DEFAULT_KWARGS)
    kwargs.update(parse_args(sys.argv[1:]))
    logdir = kwargs.pop('logdir', './ant')

    # load existing hyper params
    path = os.path.join(logdir, "kwargs.json")
    os.makedirs(logdir, exist_ok=True)
    if os.path.isfile(path):
        with open(path, "r") as f:
            existing_kwargs = json.load(f)
            existing_kwargs.update(kwargs)
            kwargs = existing_kwargs

    # save hyper params
    with open(path, "w") as f:
        json.dump(kwargs, f)

    # train a policy using soft actor critic
    np.random.seed(kwargs.pop('seed', 0))
    env = kwargs.pop('env', 'Ant-v2')
    train(logdir, gym.make(env), gym.make(env), **kwargs)
