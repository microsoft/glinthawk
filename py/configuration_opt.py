#!/usr/bin/env python3

import json
import logging
import math
import os
from typing import Dict, List, Tuple

import click
import numpy as np
import pandas as pd
from scipy import interpolate

from variants import get_model

logging.basicConfig(level=logging.INFO)


def load_profiles(log_dir: str) -> pd.DataFrame:
    entries = []
    for filename in os.listdir(log_dir):
        file_path = os.path.join(log_dir, filename)
        assert os.path.isfile(file_path)
        model_name, stage, ctx, token_pos, duration, batch_size = filename[:-4].split('_')
        df = pd.read_csv(file_path)
        entries.append([
            model_name, stage, ctx, token_pos, duration, batch_size, df['duration_us'].to_numpy()[1:].mean() / 1000
        ])
    return pd.DataFrame(entries, columns=["model_name", "stage", "ctx", "token_pos", "test_duration_s", "batch_size",
                                          "latency_us"])


def interpolate_profile(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    res = {}
    for stage, df_grp in df.groupby("stage"):
        x = df_grp['batch_size'].to_numpy()
        assert x.min() == 1
        y = df_grp['latency_us'].to_numpy()
        linear_interp = interpolate.interp1d(x, y)
        x_full = np.arange(x.min(), x.max() + 1)
        y_full = linear_interp(x_full)
        res[stage] = y_full
    return res


def get_throughput(pipe_step: float, pipe_times: List[float], serial: List[bool], in_flight: int) -> Tuple[float, float]:
    last_start = pipe_step * (in_flight - 1)
    round_2_start = sum(pipe_times)
    last_loc = last_start
    round_2_loc = round_2_start
    for i in range(len(pipe_times)):
        last_loc += pipe_times[i]
        if serial[i]:
            round_2_loc = max(last_loc, round_2_loc)
        round_2_loc += pipe_times[i]
    return in_flight / (round_2_loc - round_2_start), round_2_loc - round_2_start


def opt_single_tier(tier_1_logs: str, opt_config: str, model_name: str, opt_output: str):
    config = json.loads(opt_config)
    tier_1_profile = load_profiles(tier_1_logs)
    tier_1_full_profile = interpolate_profile(tier_1_profile)
    model = get_model(model_name, 2)

    df_data = []

    for k in range(1, config["tier_1"]["num"] + 1):
        # Get number of layers hosted per each node, the first and last layer are important because:
        #   1. The first layer hosts the embedding table
        #   2. The last layer hosts the classification table and the classification compute
        layers_per_node = math.ceil(model.n_layers / k)
        last_node_layers = model.n_layers - layers_per_node * (k - 1)

        # We might have too many nodes, e.g. 32 layers and 40 nodes. Ignore those cases
        if last_node_layers < 0:
            continue

        # Calculate the remaining memory in last and first layer
        mem_first_node = (config["tier_1"]["mem_GiB"] * 2 ** 30 -
                          model.base_size(first_layer_pre=True, last_layer_cls=k == 1, att=True) -
                          model.layer_size(pre=True, post=True) * layers_per_node)
        mem_last_node = (config["tier_1"]["mem_GiB"] * 2 ** 30 -
                         model.base_size(first_layer_pre=k == 1, last_layer_cls=True, att=True) -
                         model.layer_size(pre=True, post=True) * last_node_layers)

        # Calculate how many prompts we have KV for across all nodes (hence the min)
        kv_slots = min(mem_last_node // model.kv_size(last_node_layers),
                       mem_first_node // model.kv_size(layers_per_node))

        # Can't fit the weights on these many nodes
        if mem_last_node < 0 or mem_first_node < 0:
            continue

        # Shorthands
        rtt_ms = config['tier_1']['rtt_ms']
        cap_bps = config['tier_1']['cap_bps']

        # Loop for all possible batch sizes up to 512
        for t1_b in range(1, 512 + 1):
            # Calculate how many in_flight batches we have
            in_flight = kv_slots // t1_b

            # If we cannot load a full batch, ignore this configuration
            if in_flight == 0:
                continue

            # Three important timings here:
            #   1. The compute time for all stages but classification
            mid_step_comp = tier_1_full_profile["all_no_cls"][t1_b].item()
            #   2. The compute time for all stages
            last_step_comp = tier_1_full_profile["all"][t1_b].item()
            #   3. The commute time for BIS
            mid_step_comm = model.bis_size(t1_b, "post") * 8 / cap_bps + 1000
            assert model.bis_size(t1_b, "cls") == 0
            # The pipeline step is the maximum of:
            #   1. The compute time for nodes but the last one
            #   2. The compute time for the last nodes
            #   3. The commute time for BIS
            pipeline_step = max(
                mid_step_comp * layers_per_node,
                last_step_comp + mid_step_comp * (last_node_layers - 1),
                mid_step_comm
            )

            # Here we flesh out the full pipeline timings, including compute, commute and RTT
            #   Why separate RTT and commute (transit time due to link BW)?
            #   Because when a link is busy sending a packet due to limited bandwidth, it can't accept any other
            #   packets. When a link is busy sending a packet due to RTT latency, it can actually send other packets in
            #   parallel. So time losses due to RTT are just "delay boxes" in the pipeline, whereas time losses due to
            #   link BW should be treated like a serial part of the pipeline, similar to compute kernels.
            pipes = [mid_step_comp * layers_per_node, mid_step_comm, rtt_ms] * (k - 1)
            pipes += [mid_step_comp * (last_node_layers - 1) + last_step_comp, rtt_ms]
            serials = [True, True, False] * (k - 1) + [True, False]
            thr, tpt = get_throughput(pipeline_step, pipes, serials, in_flight)

            # Compute time is how much time a token spends in compute.
            comp_time = mid_step_comp * model.n_layers - 1 + last_step_comp
            # Commute time is how much time a token spends in network.
            # Time per token minus the two above is queueing time.
            comm_time = mid_step_comm * (k - 1) + rtt_ms * k
            cost = k * config['tier_1']['cost']

            df_data.append([
                k,
                kv_slots,
                t1_b,
                in_flight,
                thr,
                tpt,
                comp_time,
                comm_time,
                tpt - comp_time - comm_time,
                cost,
            ])

    df = pd.DataFrame(df_data,
                      columns=['t1_nodes', 't1_slots', 't1_batch_size', 'in_flight', 'throughput', 'time_per_token',
                               'compute_time', 'communication_time', 'queue_time', 'cost'])
    os.makedirs(os.path.dirname(opt_output), exist_ok=True)
    df.to_csv(opt_output)


def opt_two_tier(tier_1_logs: str, tier_2_logs: str, opt_config: str, model_name: str, opt_output: str):
    os.makedirs(opt_output, exist_ok=False)
    config = json.loads(opt_config)
    pass


@click.command()
@click.option("--tier-logs", required=True, multiple=True, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--opt-config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--model-name", required=True, type=str)
@click.option("--opt-output", required=True, type=click.Path(exists=False))
def main(**kwargs):
    if len(kwargs.get("tier_logs")) == 1:
        opt_single_tier(kwargs.get("tier_logs")[0], kwargs.get("opt_config"), kwargs.get("model_name"),
                        kwargs.get("opt_output"))
    elif len(kwargs.get("tier_logs")) == 2:
        opt_two_tier(kwargs.get("tier_logs")[0], kwargs.get("tier_logs")[1], kwargs.get("opt_config"),
                     kwargs.get("model_name"), kwargs.get("opt_output"))
    else:
        raise ValueError(f'Current cannot support {len(kwargs.get("tier_logs"))} tiers.')


if __name__ == "__main__":
    main()
