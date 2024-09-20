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
from simulation.pipe_sim import single_tier_pipeline_py, two_tier_pipeline_py
from tqdm.auto import trange

from pipeline_simulation import single_tier_pipeline, two_tier_pipeline
from variants import get_model

logging.basicConfig(level=logging.INFO)


def load_profiles(log_dir: str, model_name_match: str) -> pd.DataFrame:
    entries = []
    for filename in os.listdir(log_dir):
        file_path = os.path.join(log_dir, filename)
        assert os.path.isfile(file_path)
        model_name, stage, ctx, token_pos, duration, batch_size = filename[:-4].replace("all_no_cls",
                                                                                        "all-no-cls").split('_')
        if model_name_match != model_name:
            continue
        df = pd.read_csv(file_path, skiprows=1)
        entries.append([
            model_name, stage, ctx, int(token_pos), float(duration), int(batch_size),
            df['duration_us'].to_numpy()[1:].mean() / 1000
        ])
    return pd.DataFrame(entries, columns=["model_name", "stage", "ctx", "token_pos", "test_duration_s", "batch_size",
                                          "latency_us"])


def interpolate_profile(df: pd.DataFrame, seq_len_rescale: float) -> Dict[str, np.ndarray]:
    res = {}
    for stage, df_grp in df.groupby("stage"):
        x = df_grp['batch_size'].to_numpy()
        assert x.min() == 1, (x.min(), stage, x, df_grp)
        y = df_grp['latency_us'].to_numpy()
        linear_interp = interpolate.interp1d(x, y)
        x_full = np.arange(x.min(), x.max() + 1)
        y_full = linear_interp(x_full)
        res[stage] = np.r_[0, y_full]
    res['att'] /= seq_len_rescale
    return res


def get_throughput(pipe_step: float, pipe_times: List[float], serial: List[bool], in_flight: int) -> Tuple[
    float, float]:
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


def opt_single_tier(tier_1_logs: str, opt_config: str, model_name: str, opt_output: str, paged_kv_factor: float,
                    seq_len_rescale: float):
    with open(opt_config, "r") as f:
        config = json.load(f)
    df_t1 = load_profiles(tier_1_logs, model_name)
    t1_profiles = interpolate_profile(df_t1, seq_len_rescale)
    assert tier_1_logs.strip('/')[-4:] in ['fp16', 'fp32', 'bf16']
    model = get_model(model_name, int(tier_1_logs.strip('/')[-2:])//8)

    df_data = []

    for k in trange(1, config["tier_1"]["num"] + 1):
        # Get number of layers hosted per each node, the first and last layer are important because:
        #   1. The first layer hosts the embedding table
        #   2. The last layer hosts the classification table and the classification compute
        layers_per_node = math.ceil(model.n_layers / k)
        last_node_layers = model.n_layers - layers_per_node * (k - 1)

        # We might have too many nodes, e.g. 32 layers and 40 nodes. Ignore those cases
        if last_node_layers <= 0:
            continue

        # Calculate the remaining memory in last and first layer
        mem_first_node = (config["tier_1"]["mem_GiB"] * 2 ** 30 -
                          model.base_size(first_layer_pre=True, last_layer_cls=k == 1, att=True) -
                          model.layer_size(pre=True, post=True) * layers_per_node)
        mem_last_node = (config["tier_1"]["mem_GiB"] * 2 ** 30 -
                         model.base_size(first_layer_pre=k == 1, last_layer_cls=True, att=True) -
                         model.layer_size(pre=True, post=True) * last_node_layers)

        # Calculate how many prompts we have KV for across all nodes (hence the min)
        kv_slots = min(mem_last_node // (model.kv_size(last_node_layers) / paged_kv_factor),
                       mem_first_node // (model.kv_size(layers_per_node) / paged_kv_factor))
        kv_slots = int(kv_slots)

        # Can't fit the weights on these many nodes
        if mem_last_node < 0 or mem_first_node < 0:
            continue

        # Shorthands
        rtt_ms = config['tier_1']['rtt_ms']
        cap_bps = config['tier_1']['cap_Gbps'] * 1e9

        # Loop for all possible batch sizes up to 512
        for t1_b in trange(1, 512 + 1, leave=False):
            # Calculate how many in_flight batches we have
            in_flight = kv_slots // t1_b

            # If we cannot load a full batch, ignore this configuration
            if in_flight == 0:
                continue

            # Three important timings here:
            #   1. The compute time for all stages but classification
            mid_step_comp = t1_profiles["all-no-cls"][t1_b].item()
            #   2. The compute time for all stages
            last_step_comp = t1_profiles["all"][t1_b].item()
            #   3. The commute time for BIS
            mid_step_comm = model.bis_size(t1_b, "post") * 8 / cap_bps * 1000
            assert model.bis_size(t1_b, "cls") == 0

            # token_times = single_tier_pipeline(
            #     [layers_per_node] * (k - 1) + [last_node_layers],
            #     {
            #         "mid_layer_comp": mid_step_comp,
            #         "mid_layer_comm": mid_step_comm,
            #         "rtt": rtt_ms,
            #         "last_layer_comp": last_step_comp,
            #         "last_layer_comm": 0
            #     },
            #     in_flight
            # )
            token_times = single_tier_pipeline_py(
                [layers_per_node] * (k - 1) + [last_node_layers],
                in_flight,
                5,
                mid_step_comp,
                last_step_comp,
                mid_step_comm,
                rtt_ms / 2,
            )

            # assert np.allclose(token_times, c_token_times), (token_times, c_token_times)

            tpt = token_times[-1] - token_times[-2]
            thr = in_flight * t1_b / tpt * 1000

            # Compute time is how much time a token spends in compute.
            comp_time = mid_step_comp * (model.n_layers - 1) + last_step_comp
            # Commute time is how much time a token spends in network.
            # Time per token minus the two above is queueing time.
            comm_time = mid_step_comm * (k - 1) + rtt_ms / 2 * k
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


def opt_two_tier(tier_1_logs: str, tier_2_logs: str, opt_config: str, model_name: str, opt_output: str,
                 paged_kv_factor: float, seq_len_rescale: float):
    with open(opt_config, "r") as f:
        config = json.load(f)
    df_t1 = load_profiles(tier_1_logs, model_name)
    t1_profiles = interpolate_profile(df_t1, seq_len_rescale)
    df_t2 = load_profiles(tier_2_logs, model_name)
    t2_profiles = interpolate_profile(df_t2, seq_len_rescale)
    assert tier_1_logs.strip('/')[-4:] in ['fp16', 'fp32', 'bf16']
    model_t1 = get_model(model_name, int(tier_1_logs.strip('/')[-2:])//8)
    assert tier_2_logs.strip('/')[-4:] in ['fp16', 'fp32', 'bf16']
    model_t2 = get_model(model_name, int(tier_2_logs.strip('/')[-2:])//8)

    df_data = []

    for k in trange(1, config["tier_1"]["num"] + 1):
        for d in trange(1, config["tier_2"]["num"] + 1, leave=False):
            # Get number of layers hosted per each node, the first and last layer are important because:
            #   1. The first layer hosts the embedding table
            #   2. The last layer hosts the classification table and the classification compute
            layers_per_node = math.ceil(model_t1.n_layers / k)
            last_node_layers = model_t1.n_layers - layers_per_node * (k - 1)

            # We might have too many nodes, e.g. 32 layers and 40 nodes. Ignore those cases
            if last_node_layers <= 0:
                continue

            # Calculate the remaining memory in last and first layer
            mem_t1_first_node = (config["tier_1"]["mem_GiB"] * 2 ** 30 -
                                 model_t1.base_size(first_layer_pre=True, last_layer_cls=k == 1, att=True) -
                                 model_t1.layer_size(pre=True, post=True) * layers_per_node)
            mem_t1_last_node = (config["tier_1"]["mem_GiB"] * 2 ** 30 -
                                model_t1.base_size(first_layer_pre=k == 1, last_layer_cls=True, att=True) -
                                model_t1.layer_size(pre=True, post=True) * last_node_layers)
            mem_t2_node = (config["tier_2"]["mem_GiB"] * 2 ** 30 - model_t2.base_size(att=True))

            # Calculate how many prompts we have KV for across all nodes (hence the min)
            kv_slots_t1 = min(mem_t1_last_node // (model_t1.kv_size(last_node_layers) / paged_kv_factor),
                              mem_t1_first_node // (model_t1.kv_size(layers_per_node) / paged_kv_factor))
            kv_slots_t2 = mem_t2_node // (model_t2.kv_size(max(last_node_layers, layers_per_node)) / paged_kv_factor)

            kv_slots_t1 = int(kv_slots_t1)
            kv_slots_t2 = int(kv_slots_t2)

            # Can't fit the weights on these many nodes
            if mem_t1_last_node < 0 or mem_t1_first_node < 0 or mem_t2_node < 0 or kv_slots_t2 == 0:
                continue

            # Shorthands
            rtt_11_ms = config['tier_1']['rtt_ms']
            cap_11_bps = config['tier_1']['cap_Gbps'] * 1e9
            rtt_12_ms = config['inter_tier_rtt_ms']
            cap_12_bps = config['inter_tier_cap_Gbps'] * 1e9

            max_t2_b = min((t1_profiles["pre"].shape[0] - 1) // d, t2_profiles["att"].shape[0] - 1)

            # Loop for all possible batch sizes for tier 2
            for t2_b in trange(1, max_t2_b + 1, leave=False):
                # Calculate how many in_flight batches we have
                in_flight = kv_slots_t2 // t2_b

                # If we cannot load a full batch, ignore this configuration
                if in_flight == 0:
                    continue

                t1_att_b = kv_slots_t1 // in_flight
                t1_b = t2_b * d + t1_att_b

                if t1_b >= t1_profiles["pre"].shape[0]:
                    continue
                if t1_b >= t1_profiles["post"].shape[0]:
                    continue
                if t1_b >= t1_profiles["cls"].shape[0]:
                    continue
                if t1_att_b >= t1_profiles["att"].shape[0]:
                    continue

                # Four important timings here:
                #   1. The compute time for tier 1 besides classification
                mid_step_t1_comp = t1_profiles["pre"][t1_b].item() + t1_profiles["att"][t1_att_b].item() + \
                                   t1_profiles["post"][t1_b].item()
                #   2. The compute time for tier 1, all stages
                last_step_t1_comp = t1_profiles["pre"][t1_b].item() + t1_profiles["att"][t1_att_b].item() + \
                                    t1_profiles["post"][t1_b].item() + t1_profiles["cls"][t1_b].item()
                #   3. The compute time for tier 2
                t2_comp = t2_profiles["att"][t2_b].item()
                #   4. The commute time for BIS in tier 1
                mid_step_t1_comm = model_t1.bis_size(t1_b, "post") * 8 / cap_11_bps * 1000
                #   5. The commute time for BIS in tier 2
                mid_step_t2_comm = model_t1.bis_size(t2_b * d, "pre") * 8 / cap_12_bps * 1000
                #   5. The commute time for BIS in tier 2
                mid_step_t2_comm_back = model_t1.bis_size(t2_b * d, "att") * 8 / cap_12_bps * 1000

                # There are 6 active stages
                max_in_flight_needed = k * (1 + math.ceil(
                    (mid_step_t1_comm + t2_comp + mid_step_t2_comm_back + rtt_12_ms) / mid_step_t1_comp))
                if in_flight > max_in_flight_needed * 2:
                    continue

                # token_times = two_tier_pipeline(
                #     [layers_per_node] * (k - 1) + [last_node_layers],
                #     {
                #         "mid_layer_t1": mid_step_t1_comp,
                #         "t1_to_t2_comm": mid_step_t2_comm,
                #         "t1_to_t2_rtt": rtt_12_ms,
                #         "t2_comp": t2_comp,
                #         "t2_to_t1_comm": mid_step_t2_comm_back,
                #         "last_layer_t1": last_step_t1_comp,
                #     },
                #     in_flight
                # )

                token_times = two_tier_pipeline_py(
                    [layers_per_node] * (k - 1) + [last_node_layers],
                    in_flight,
                    2,
                    mid_step_t1_comp,
                    last_step_t1_comp,
                    t2_comp,
                    mid_step_t2_comm,
                    rtt_12_ms / 2,
                    mid_step_t2_comm_back,
                    rtt_12_ms / 2
                    )

                # assert np.allclose(token_times, c_token_times), (token_times, c_token_times)

                tpt = token_times[-1] - token_times[-2]
                thr = in_flight * t1_b / tpt * 1000

                # Compute time is how much time a token spends in compute.
                comp_time = (t2_comp + mid_step_t1_comp) * (model_t1.n_layers - 1) + (last_step_t1_comp + t2_comp)
                # Commute time is how much time a token spends in network.
                # Time per token minus the two above is queueing time.
                comm_time = (mid_step_t2_comm * model_t1.n_layers +
                             mid_step_t2_comm_back * model_t1.n_layers +
                             mid_step_t1_comm * (k - 1) +
                             rtt_12_ms * model_t1.n_layers +
                             rtt_11_ms / 2 * k)
                cost = k * config['tier_1']['cost'] + d * k * config['tier_2']['cost']

                t1_to_t2_bw = model_t1.bis_size(t2_b * d, "pre") * layers_per_node * in_flight / tpt * 1000 * 8
                t2_to_t1_bw = model_t1.bis_size(t2_b * d, "att") * layers_per_node * in_flight / tpt * 1000 * 8

                df_data.append([
                    k,
                    d,
                    kv_slots_t1,
                    kv_slots_t2,
                    t1_b,
                    t1_att_b,
                    t2_b,
                    in_flight,
                    thr,
                    t1_to_t2_bw,
                    t2_to_t1_bw,
                    tpt,
                    comp_time,
                    comm_time,
                    tpt - comp_time - comm_time,
                    cost,
                ])

        df = pd.DataFrame(df_data,
                          columns=['t1_nodes', 't2_per_t1_nodes', 't1_slots', 't2_slots', 't1_batch_size',
                                   't1_att_batch_size', 't2_batch_size', 'in_flight', 'throughput', 't1_to_t2_bw',
                                   't2_to_t1_bw', 'time_per_token', 'compute_time', 'communication_time', 'queue_time',
                                   'cost'])
        os.makedirs(os.path.dirname(opt_output), exist_ok=True)
        df.to_csv(opt_output)


@click.command()
@click.option("--tier-logs", required=True, multiple=True, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--opt-config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--model-name", required=True, type=str)
@click.option("--paged-kv-reuse-factor", required=False, default=3, type=float)
@click.option("--seq-len-rescale", required=False, default=3, type=float)
@click.option("--opt-output", required=True, type=click.Path(exists=False))
def main(**kwargs):
    if len(kwargs.get("tier_logs")) == 1:
        opt_single_tier(kwargs.get("tier_logs")[0], kwargs.get("opt_config"), kwargs.get("model_name"),
                        kwargs.get("opt_output"), kwargs.get("paged_kv_reuse_factor"), kwargs.get("seq_len_rescale"))
    elif len(kwargs.get("tier_logs")) == 2:
        opt_two_tier(kwargs.get("tier_logs")[0], kwargs.get("tier_logs")[1], kwargs.get("opt_config"),
                     kwargs.get("model_name"), kwargs.get("opt_output"), kwargs.get("paged_kv_reuse_factor"),
                     kwargs.get("seq_len_rescale"))
    else:
        raise ValueError(f'Current cannot support {len(kwargs.get("tier_logs"))} tiers.')


if __name__ == "__main__":
    main()
