#!/usr/bin/env python3

import matplotlib.pyplot as plt
import click
import pandas as pd
from util import nice_ax, nice_plt_cfg

PAGED_KV_REUSE_FACTOR = 2.5


def plot(single_tier_sim: str, two_tier_sim: str, plot_dir: str):
    # columns=['t1_nodes', 't1_slots', 't1_batch_size', 'in_flight', 'throughput', 'time_per_token', 'compute_time',
    # 'communication_time', 'queue_time', 'cost']
    df_single = pd.read_csv(single_tier_sim)

    # columns=['t1_nodes', 't2_per_t1_nodes', 't1_slots', 't2_slots', 't1_batch_size', 't1_att_batch_size',
    # 't2_batch_size', 'in_flight', 'throughput', 'time_per_token', 'compute_time', 'communication_time', 'queue_time',
    # 'cost']
    df_double = pd.read_csv(two_tier_sim)

    nice_plt_cfg()
    fig, ax = plt.subplots(1, 1, figsize=(3.25, 2))

    df_single_best = df_single.loc[df_single.groupby(['t1_nodes'])['throughput'].idxmax()].reset_index(drop=True)
    df_single_best['cost_per_thr'] = df_single_best['cost'] / df_single_best['throughput']
    print(df_single_best)

    df_double_best = df_double.loc[df_double.groupby(['t1_nodes', "t2_per_t1_nodes"])['throughput'].idxmax()].reset_index(drop=True)
    df_double_best['cost_per_thr'] = df_double_best['cost'] / df_double_best['throughput']
    df_double_best = df_double_best.loc[df_double_best.groupby(["t1_nodes"])['cost_per_thr'].idxmin()].reset_index(drop=True)
    print(df_double_best)

    ax.scatter(df_single_best["cost"]/1000, df_single_best["throughput"]/1000, color='C3', label='Single-Tier', marker='v', s=1)
    ax.scatter(df_double_best["cost"]/1000, df_double_best["throughput"]/1000, color='C2', label='Two-Tier', marker='s', s=1)

    ax.set_xlabel(r"Cost (\$K)")
    ax.set_ylabel(r"Throughput (Ktk/s)")

    nice_ax(ax)
    ax.legend(ncol=2, loc='lower left', bbox_to_anchor=(0.1, 1.03, 0.8, 0.1), mode="expand", handlelength=3)
    fig.tight_layout()

    plt.savefig(f"{plot_dir}/config_simulation.pdf", format='pdf')


@click.command()
@click.option("--single-tier-sim", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--two-tier-sim", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--plot-dir", required=True, type=click.Path(exists=True, dir_okay=True, file_okay=False))
def main(**kwargs):
    plot(**kwargs)


if __name__ == "__main__":
    main()
