import click


@click.command()
@click.option('--logging-dir', type=str)
@click.option('--tag', type=str)
@click.option('--xlabel', type=str)
@click.option('--ylabel', type=str)
def plot(logging_dir, tag, xlabel, ylabel):

    from collections import defaultdict
    import glob
    import os
    import pandas as pd
    import tensorflow as tf
    import tqdm
    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt
    import json

    plt.rcParams['text.usetex'] = False
    matplotlib.rc('font', family='serif', serif='cm10')
    matplotlib.rc('mathtext', fontset='cm')
    color_palette = ['#EE7733',
                     '#0077BB',
                     '#33BBEE',
                     '#009988',
                     '#CC3311',
                     '#EE3377',
                     '#BBBBBB',
                     '#000000']
    palette = sns.color_palette(color_palette)
    sns.palplot(palette)
    sns.set_palette(palette)

    def pretty(s):
        return s.replace('_', ' ').title()

    # get the experiment folders
    dirs = glob.glob(os.path.join(logging_dir, "trial-*"))

    # get the hyper parameters for each experiment
    params = []
    for d in dirs:
        with open(os.path.join(d, 'params.json'), 'r') as f:
            params.append(json.load(f))

    # concatenate all params along axis 1
    all_params = defaultdict(list)
    for p in params:
        for key, val in p.items():
            if val not in all_params[key]:
                all_params[key].append(val)

    # locate the params of variation in this experiment
    params_of_variation = []
    for key, val in all_params.items():
        if len(val) > 1 and (not isinstance(val[0], dict)
                             or 'seed' not in val[0]):
            params_of_variation.append(key)

    # get the task and algorithm name
    task_name = params[0]['eval_env']
    if len(params_of_variation) == 0:
        params_of_variation.append('eval_env')

    # read data from tensor board
    data = pd.DataFrame(columns=[xlabel, ylabel] + params_of_variation)
    for d, p in tqdm.tqdm(zip(dirs, params)):
        for f in glob.glob(os.path.join(d, '*/events.out*')):
            for e in tf.compat.v1.train.summary_iterator(f):
                for v in e.summary.value:
                    if v.tag == tag:
                        row = {ylabel: tf.make_ndarray(
                            v.tensor).tolist(), xlabel: e.step}
                        for key in params_of_variation:
                            row[key] = f'{pretty(key)} = {p[key]}'
                        data = data.append(row, ignore_index=True)

    # save a separate plot for every hyper parameter
    for key in params_of_variation:
        plt.clf()
        g = sns.relplot(x=xlabel, y=ylabel, hue=key, data=data,
                        kind="line", height=5, aspect=1.5,
                        facet_kws={"legend_out": True})
        g.set(title=f'{task_name}')
        plt.savefig(f'{task_name}_{key}_{tag.replace("/", "_")}.png',
                    bbox_inches='tight')


if __name__ == "__main__":
    plot()
