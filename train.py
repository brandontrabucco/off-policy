from offpolicy import train
import gym
import sys


def clean_arguments(args):
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
        # single hyphen, no arg
        elif a.startswith('-') and (not b or b.startswith('-')):
            val = True

        # double hypen, arg
        # single hypen, arg
        elif a.startswith('-') and b and not b.startswith('-'):
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

    kwargs = clean_arguments(sys.argv[1:])
    train(kwargs.pop('logdir', './ant_sac'),
          gym.make(kwargs.pop('env', 'Ant-v2')),
          gym.make(kwargs.pop('env', 'Ant-v2')),
          kwargs.pop('alg', 'SAC'),
          **kwargs)
