from collections.abc import Callable
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from simulator.simulator_model import MMDiT, Transformer

logger = logging.getLogger(__name__)


@torch.no_grad()
def run_benchmarking(
    seq_len,
    d_token,
    d_head,
    model: Callable,
    num_warmup=5,
    num_repeat=5,
):
    from utils.transformer_utils import get_rope_cossin

    device = torch.cuda.current_device()

    seq_lens = [seq_len]
    rope_cossin = get_rope_cossin(seq_lens, d_head, device=device)

    output_sum = 0.0
    for _ in range(num_warmup):
        input_data = torch.randn(seq_len, d_token, device=device, dtype=torch.bfloat16)
        _, output_data = model(input_data, seq_lens, [rope_cossin])

        output_sum += output_data.mean().item()
    torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(num_repeat):
        input_data = torch.randn(seq_len, d_token, device=device, dtype=torch.bfloat16)
        _, output_data = model(input_data, seq_lens, [rope_cossin])

        output_sum += output_data.mean().item()
    torch.cuda.synchronize()
    end_time = time.time()

    latency = (end_time - start_time) / num_repeat
    return latency


class WorkloadEstimator(nn.Module):
    def __init__(self):
        super().__init__()

    def benchmark_workload(self, model: Callable) -> float:
        pass

    def predict_workload(self, seq_len: int) -> float:
        pass


class TheoreticalWorkloadEstimator(WorkloadEstimator):
    def __init__(
        self,
        seq_len_test_range: tuple[int, int] = (448, 65536),
        seq_len_test_pow_base: int | None = None,
        seg_n_bins: int | None = 10,
        d_token: int = 4096,
        gamma_correction: bool = True,
    ):
        """
        The TheoreticalWorkloadEstimator is a workload estimator that uses the theoretical workload of the model.
        gamma_correction is a boolean that determines whether to use the gamma correction.
        """
        super().__init__()
        self.gamma_correction = gamma_correction
        self.d_token = d_token
        if seq_len_test_pow_base is not None:
            pow_low = math.log(seq_len_test_range[0], seq_len_test_pow_base)
            pow_high = math.log(seq_len_test_range[1], seq_len_test_pow_base)

            self.seq_len_test_list = [
                int(seq_len_test_pow_base**i) for i in range(math.floor(pow_low), math.ceil(pow_high) + 1)
            ]
        else:
            assert seg_n_bins is not None
            self.seq_len_test_list = (
                np.linspace(seq_len_test_range[0], seq_len_test_range[1], seg_n_bins).astype(np.int32).tolist()
            )

    def benchmark_workload(
        self,
        model: Callable,
        d_head: int,
        test_latencies: torch.Tensor | None = None,
        test_seq_lens: torch.Tensor | None = None,
    ) -> float:
        """
        If test_latencies and test_seq_lens are not provided, the benchmark_workload will be run on the seq_len_test_list.
        """
        device = torch.cuda.current_device()
        if test_latencies is None:
            test_latencies = []
            for seq_len in self.seq_len_test_list:
                latency = run_benchmarking(seq_len, d_token=self.d_token, d_head=d_head, model=model)
                test_latencies.append(latency)
            test_latencies = torch.tensor(test_latencies, dtype=torch.float32, device=device)
            test_seq_lens = torch.tensor(self.seq_len_test_list, dtype=torch.int32, device=device)

        self.register_buffer("test_seq_lens", test_seq_lens.clone())
        self.register_buffer("test_latencies", test_latencies.clone())

        if self.gamma_correction:
            # Build equations: latency = gamma_a * seq_len * d_token**2 + gamma_b * seq_len^2 * d_token
            A = torch.zeros(len(self.seq_len_test_list), 2, device=device, dtype=torch.float32)
            b = torch.zeros(len(self.seq_len_test_list), device=device, dtype=torch.float32)
            A[:, 0] = 24.0 * test_seq_lens * self.d_token**2
            A[:, 1] = 4.0 * test_seq_lens * test_seq_lens * self.d_token
            b = test_latencies

            # Solve for both k and gamma
            k, k_times_gamma = torch.linalg.lstsq(A, b).solution
            # The actual flops theoretical equation is 24 * seq_len * d_token^2 + 4 * gamma * seq_len^2 * d_token
            # The latency is approximately k * flops
            gamma = k_times_gamma / k
            self.register_buffer("gamma", gamma)
            self.register_buffer("k", k)
        else:
            # Solve for k only
            A = torch.zeros(len(self.seq_len_test_list), 1, device=device, dtype=torch.float32)
            b = torch.zeros(len(self.seq_len_test_list), device=device, dtype=torch.float32)
            A[:, 0] = 24.0 * test_seq_lens * self.d_token**2 + 4.0 * test_seq_lens * test_seq_lens * self.d_token
            b = test_latencies
            k = torch.linalg.lstsq(A, b).solution
            self.register_buffer("k", k)

    def predict_workload(self, seq_len: int) -> float:
        if self.gamma_correction:
            return (self.k * (24.0 * seq_len * self.d_token**2 + 4.0 * self.gamma * seq_len**2 * self.d_token)).item()
        else:
            return (self.k * (24.0 * seq_len * self.d_token**2 + 4.0 * seq_len**2 * self.d_token)).item()


def plot_workload_comparison(
    estimator_corrected,
    estimator_naive,
    predicted_latencies_corrected,
    predicted_latencies_naive,
    output_file,
):
    import plotly.graph_objects as go

    # Create the plot
    fig = go.Figure()

    # Add scatter plot for actual data
    fig.add_trace(
        go.Scatter(
            x=estimator_corrected.test_seq_lens.cpu().numpy(),
            y=estimator_corrected.test_latencies.cpu().numpy(),
            mode="markers",
            name="Actual",
            marker={"color": "orange", "size": 8},
            hovertemplate="<b>Sequence Length:</b> %{x}<br><b>Latency:</b> %{y:.4f}<extra></extra>",
        )
    )

    # Add line plot for estimated data
    fig.add_trace(
        go.Scatter(
            x=estimator_corrected.test_seq_lens.cpu().numpy(),
            y=np.array(predicted_latencies_corrected),
            mode="lines",
            name="w/ gamma correction",
            line={"color": "blue", "width": 2},
            hovertemplate="<b>Sequence Length:</b> %{x}<br><b>Estimated Latency:</b> %{y:.4f}<extra></extra>",
        )
    )

    # Add line plot for naive estimated data
    fig.add_trace(
        go.Scatter(
            x=estimator_naive.test_seq_lens.cpu().numpy(),
            y=np.array(predicted_latencies_naive),
            mode="lines",
            name="w/o gamma correction",
            line={"color": "green", "width": 2},
            hovertemplate="<b>Sequence Length:</b> %{x}<br><b>Naive Estimated Latency:</b> %{y:.4f}<extra></extra>",
        )
    )

    # Update layout
    fig.update_layout(
        title="Theoretical Workload Estimator Comparison",
        xaxis_title="Sequence Length",
        yaxis_title="Normalized Latency",
        xaxis_type="linear",
        yaxis_type="linear",
        hovermode="x unified",
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
        template="plotly_white",
    )

    folder = os.path.dirname(output_file)
    os.makedirs(folder, exist_ok=True)
    # Save as PNG
    fig.write_image(f"{folder}/theoretical_workload_estimator.png")
    # Save as SVG for vector graphics
    fig.write_image(f"{folder}/theoretical_workload_estimator.svg", format="svg")


if __name__ == "__main__":
    """
    python workload_estimator.py [output_file] [--d_model d_model] [--d_head d_head] [--causal causal]
    [--use_flux use_flux] [--n_ds_layers n_ds_layers] [--n_ss_layers n_ss_layers]
    This script will benchmark the workload of the model and estimate
    The workload using the theoretical workload estimator, and plot the results.
    If `output_file` is provided, run `plotly_get_chrome` to install the plotly chrome extension before running this script.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Workload Estimator")
    parser.add_argument("output_file", nargs="?", help="Output file to write gamma value to")
    parser.add_argument("--d_model", type=int, default=3072, help="Model dimension")
    parser.add_argument("--d_head", type=int, default=128, help="Head dimension")
    parser.add_argument("--causal", type=int, default=0, help="Use causal attention")
    # flux specific
    parser.add_argument("--use_flux", type=int, default=0, help="Use flux")
    parser.add_argument("--n_ds_layers", type=int, default=19, help="Number of double stream layers")
    parser.add_argument("--n_ss_layers", type=int, default=38, help="Number of single stream layers")
    args = parser.parse_args()

    device = torch.cuda.current_device()
    d_model = args.d_model
    d_head = args.d_head

    if args.use_flux:
        model = MMDiT(d_model=d_model, d_head=d_head, n_ds_layers=args.n_ds_layers, n_ss_layers=args.n_ss_layers)
    else:
        model = Transformer(d_model=d_model, d_head=d_head, n_blocks=1, causal=bool(args.causal))

    model = model.to(device=device, dtype=torch.bfloat16)

    logger.info(f"Gamma_correction=True, Benchmarking workload for model with d_model={d_model}, d_head={d_head}")
    estimator_corrected = TheoreticalWorkloadEstimator(d_token=d_model, gamma_correction=True).to(device)
    estimator_corrected.benchmark_workload(model, d_head=d_head)
    logger.info(f"Estimated gamma: {estimator_corrected.gamma.item()}")

    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(f"{estimator_corrected.gamma.item()}")

    logger.info(f"Gamma_correction=False, Benchmarking workload for model with d_model={d_model}, d_head={d_head}")
    estimator_naive = TheoreticalWorkloadEstimator(d_token=d_model, gamma_correction=False).to(device)
    estimator_naive.benchmark_workload(
        model,
        d_head=d_head,
        test_latencies=estimator_corrected.test_latencies,
        test_seq_lens=estimator_corrected.test_seq_lens,
    )

    predicted_latencies_corrected = []
    predicted_latencies_naive = []
    for seq_len in estimator_corrected.test_seq_lens.cpu().numpy().tolist():
        predicted_latency_corrected = estimator_corrected.predict_workload(seq_len)
        predicted_latencies_corrected.append(predicted_latency_corrected)
        predicted_latency_naive = estimator_naive.predict_workload(seq_len)
        predicted_latencies_naive.append(predicted_latency_naive)

    # Plot the results
    if args.output_file:
        plot_workload_comparison(
            estimator_corrected,
            estimator_naive,
            predicted_latencies_corrected,
            predicted_latencies_naive,
            args.output_file,
        )
