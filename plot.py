import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
sns.set(style="darkgrid")


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Plot")
    parser.add_argument(
        '--files',
        type=str,
        nargs='+')
    parser.add_argument(
        '--names',
        type=str,
        nargs='+')
    parser.add_argument(
        '--limit',
        type=int,
        default=2500000)
    parser.add_argument(
        '--out',
        type=str,
        default='plot.png')
    args = parser.parse_args()

    df = pd.DataFrame(columns=[
        'Wall time', 'Name', 'Environment Step', 'Average Return'])
    column_remap = {
        'Step': 'Environment Step', 'Value': 'Average Return'}

    for f, name in zip(args.files, args.names):
        df0 = pd.read_csv(f).rename(columns=column_remap)
        df0['Name'] = name
        df = pd.concat([df, df0])

    df = df[df['Environment Step'] < args.limit]

    sns.lineplot(x='Environment Step',
                 y='Average Return',
                 hue='Name',
                 data=df)
    plt.savefig(args.out)
