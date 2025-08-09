import argparse
import logging
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from knapformer import SequenceBalancer
from knapformer.simulator.simulator_data import DataSimulator
from knapformer.simulator.simulator_model import Transformer, create_mmdit, create_transformer
from knapformer.utils.fsdp_utils import setup_distributed, teardown_distributed
from knapformer.utils.perf_utils import get_peak_tflops_per_second

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


def simulate_one_step(
    data_simulator: DataSimulator,
    transformer: Transformer,
    sequence_balancer: SequenceBalancer | None = None,
    skip_bwd: bool = True,
    use_flux: bool = False,
):
    with torch.set_grad_enabled(not skip_bwd):
        packed_seq_lens, packed_seqs, packed_features = data_simulator.next_batch()
        # We make sure packed_seqs is bfloat16
        packed_seqs = packed_seqs.type(torch.bfloat16)

        num_tokens = packed_seqs.shape[0]

        if use_flux:
            vec = torch.randn(
                len(packed_seq_lens),
                packed_seqs.shape[-1],
                device=packed_seqs.device,
                dtype=packed_seqs.dtype,
            )  # (B, D)
            vec = vec.requires_grad_(True)
            seq_out_lens, seq_out = transformer(packed_seqs, packed_seq_lens, packed_features, vec, sequence_balancer)
        else:
            seq_out_lens, seq_out = transformer(packed_seqs, packed_seq_lens, packed_features, sequence_balancer)

        if not skip_bwd:
            loss = torch.mean(seq_out)
            loss.backward()

            # flush gradients
            for param in transformer.parameters():
                param.grad = None

    return num_tokens


def parse_args():
    parser = argparse.ArgumentParser(description="Sequence Balancer Simulator")
    parser.add_argument(
        "--data_codes",
        type=str,
        default="g3b16i256f1s0,g2b8i512f1s0,g2b4i1024f1s0,g1b1i2048f1s0",
        help="List of data codes for simulation (comma-separated string)",
    )
    parser.add_argument(
        "--balancer_config",
        type=str,
        default="g1n2+g2n1+g4n1",
        help="Balancer configuration string",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Gamma parameter for workload estimation",
    )
    parser.add_argument("--d_model", type=int, default=4096, help="Model dimension")
    parser.add_argument("--d_head", type=int, default=128, help="Head dimension")
    parser.add_argument("--n_layers", type=int, default=32, help="Number of layers")
    parser.add_argument("--shard_size", type=int, default=8, help="Shard size for FSDP")
    parser.add_argument("--causal", type=int, default=0, help="Use causal attention")
    # flux specific
    parser.add_argument("--use_flux", type=int, default=0, help="Use flux")
    parser.add_argument("--n_ds_layers", type=int, default=19, help="Number of double stream layers")
    parser.add_argument("--n_ss_layers", type=int, default=38, help="Number of single stream layers")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """
    torchrun --nproc_per_node=8 simulate.py \
        --data_codes "g3b16i256f1s0,g2b8i512f1s0,g2b4i1024f1s0,g1b1i2048f1s0" \
        --balancer_config g8n1 \
        --use_flux 1 \
        2>&1 | tee simulator.log
    """

    args = parse_args()

    torch.manual_seed(0)

    setup_distributed()

    data_codes = [code.strip() for code in args.data_codes.split(",")]
    balancer_config = args.balancer_config
    d_model, d_head, n_layers = args.d_model, args.d_head, args.n_layers
    shard_size = args.shard_size
    causal = bool(args.causal)
    param_dtype = torch.bfloat16
    reduce_dtype = torch.float32
    ac_freq = 1
    skip_bwd = False
    gamma = args.gamma

    use_flux = bool(args.use_flux)
    n_ds_layers = args.n_ds_layers
    n_ss_layers = args.n_ss_layers

    if dist.get_rank() == 0:
        logger.info(f"data_codes: {data_codes}")
        logger.info(f"balancer_config: {balancer_config}")
        logger.info(f"d_model: {d_model}, d_head: {d_head}, n_layers: {n_layers}")
        logger.info(
            f"shard_size: {shard_size}, param_dtype: {param_dtype}, reduce_dtype: {reduce_dtype}, ac_freq: {ac_freq}"
        )
        logger.info(f"gamma: {gamma}")
        logger.info(f"use_flux: {use_flux}, n_ds_layers: {n_ds_layers}, n_ss_layers: {n_ss_layers}")

    data_simulator = DataSimulator(
        data_codes=data_codes,
        d_model=d_model,
        d_head=d_head,
        vae_spatial_rate=16.0,
        vae_temporal_rate=3.4,  # 17/5 = 3.4
        min_aspect_ratio_multiplier=0.96,
        max_aspect_ratio_multiplier=1.04,
        random_seed=dist.get_rank(),
    )
    sequence_balancer = None
    if balancer_config is not None and len(balancer_config) > 0:
        sequence_balancer = SequenceBalancer(balancer_config, gamma=gamma)

    if use_flux:
        transformer = create_mmdit(
            d_model=d_model,
            d_head=d_head,
            n_ds_layers=n_ds_layers,
            n_ss_layers=n_ss_layers,
            shard_size=shard_size,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            ac_freq=ac_freq,
        )
    else:
        transformer = create_transformer(
            d_model=d_model,
            d_head=d_head,
            n_layers=n_layers,
            shard_size=shard_size,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            ac_freq=ac_freq,
            causal=causal,
        )

    n_params = sum(p.numel() for p in transformer.parameters())
    if dist.get_rank() == 0:
        logger.info(f"Number of parameters: {n_params}")

    n_warmup_steps, n_steps = 10, 50

    if dist.get_rank() == 0:
        logger.info(f"Warming up for {n_warmup_steps} steps")
    for _ in tqdm(range(n_warmup_steps), disable=dist.get_rank() != 0):
        simulate_one_step(
            data_simulator,
            transformer,
            sequence_balancer,
            skip_bwd=skip_bwd,
            use_flux=use_flux,
        )

    torch.cuda.synchronize()
    dist.barrier()
    if dist.get_rank() == 0:
        logger.info(f"Running for {n_steps} steps")
    if sequence_balancer is not None:
        sequence_balancer.turn_on_tracking(n_layers, not skip_bwd)
    start_time = time.time()
    num_tokens = 0
    for _ in tqdm(range(n_steps), disable=dist.get_rank() != 0):
        num_tokens += simulate_one_step(
            data_simulator,
            transformer,
            sequence_balancer,
            skip_bwd=skip_bwd,
            use_flux=use_flux,
        )
    torch.cuda.synchronize()
    dist.barrier()
    end_time = time.time()
    if dist.get_rank() == 0:
        logger.info(f"Per step time: {(end_time - start_time) / n_steps} seconds")
    num_tokens = torch.tensor(num_tokens, dtype=torch.int64, device=torch.cuda.current_device())
    dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM)
    if dist.get_rank() == 0:
        logger.info(f"Throughput on {dist.get_world_size()} GPUs: {num_tokens / (end_time - start_time)} tokens/s")
        if sequence_balancer is not None:
            sequence_balancer.print_tracking_results()

            tracked_tflops = np.sum(sequence_balancer.tracked_tflops)
            tflops_per_second = tracked_tflops / (end_time - start_time)

            peak_tflops_per_second = get_peak_tflops_per_second() * dist.get_world_size()
            logger.info(f"HFU: {tflops_per_second / peak_tflops_per_second * 100:.2f}%")

    teardown_distributed()
