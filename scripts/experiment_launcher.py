#!/usr/bin/env python3
"""
Python-based experiment launcher that generates and runs bash scripts for all experiment setups.
This replaces the complex bash script approach with a more readable and maintainable solution.
"""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
import time


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    name: str
    data_codes: str
    balancer_config: str
    d_model: int = 3072
    d_head: int = 128
    n_layers: int = 57
    shard_size: int = 8
    causal: int = 0
    use_flux: int = 0
    n_ds_layers: int = 19
    n_ss_layers: int = 38
    log_dir: str = "logs"


class ExperimentLauncher:
    """Manages generation and execution of experiment bash scripts."""

    def __init__(self, script_dir: str = "tmp/scripts", gap_seconds: int = 10):
        self.script_dir = Path(script_dir)
        self.script_dir.mkdir(parents=True, exist_ok=True)
        self.gap_seconds = gap_seconds
        self.run_timestamp = time.strftime("%Y%m%d_%H%M%S")

    def load_bash_template(self) -> str:
        """Load the bash template from external file."""
        template_path = Path(__file__).parent / "experiment_template.sh"
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")
        return template_path.read_text()

    def generate_script(self, config: ExperimentConfig) -> Path:
        """Generate a bash script for a specific experiment configuration."""
        # Load template
        template_content = self.load_bash_template()

        # Generate ARG_ variable assignments
        log_dir_path = f"{config.log_dir}/{self.run_timestamp}_{config.name}"
        arg_assignments = [
            f'ARG_LOG_DIR="{log_dir_path}"',
            f'ARG_DATA_CODES="{config.data_codes}"',
            f'ARG_BALANCER_CONFIG="{config.balancer_config}"',
            f'ARG_D_MODEL="{config.d_model}"',
            f'ARG_D_HEAD="{config.d_head}"',
            f'ARG_N_LAYERS="{config.n_layers}"',
            f'ARG_SHARD_SIZE="{config.shard_size}"',
            f'ARG_CAUSAL="{config.causal}"',
            f'ARG_USE_FLUX="{config.use_flux}"',
            f'ARG_N_DS_LAYERS="{config.n_ds_layers}"',
            f'ARG_N_SS_LAYERS="{config.n_ss_layers}"',
        ]

        # Prepend ARG_ assignments to the template
        script_content = "#!/bin/bash\n" + "\n".join(arg_assignments) + "\n\n" + template_content

        # Create separate folder for each script
        script_folder = self.script_dir / f"{self.run_timestamp}_{config.name}"
        script_folder.mkdir(parents=True, exist_ok=True)

        script_path = script_folder / f"run_{config.name}.sh"
        with open(script_path, "w") as f:
            f.write(script_content)

        # Make script executable
        os.chmod(script_path, 0o755)

        return script_path

    def aggregate_logs(self, config: ExperimentConfig) -> dict:
        """Aggregate logs using Python instead of bash."""
        log_dir_path = f"{config.log_dir}/{self.run_timestamp}_{config.name}"
        log_dir = Path(log_dir_path)

        # Read gamma value
        gamma_file = log_dir / "workload_estimator_gamma.txt"
        gamma = None
        if gamma_file.exists():
            gamma = gamma_file.read_text().strip()

        # Patterns to extract from logs
        patterns = {
            "Number of parameters": r"Number of parameters.*",
            "data_codes": r"data_codes.*",
            "balancer_config": r"balancer_config.*",
            "Per step time": r"Per step time.*",
            "Throughput on": r"Throughput on.*",
            "HFU": r"HFU.*",
            "Per GPU Capacity": r"Per GPU Capacity.*",
            "Original Imbalance Ratio": r"Original Imbalance Ratio.*",
            "Balanced Imbalance Ratio": r"Balanced Imbalance Ratio.*",
        }

        def extract_metrics_from_log(log_file: Path) -> dict[str, list[str]]:
            """Extract metrics from a log file."""
            metrics = {key: [] for key in patterns.keys()}
            if not log_file.exists():
                return metrics

            content = log_file.read_text()
            for key, pattern in patterns.items():
                matches = re.findall(pattern, content, re.IGNORECASE)
                metrics[key] = matches
            return metrics

        # Extract metrics from both runs
        with_balancer_log = log_dir / "simulator_with_balancer_0.log"
        without_balancer_log = log_dir / "simulator_without_balancer_0.log"

        with_balancer_metrics = extract_metrics_from_log(with_balancer_log)
        without_balancer_metrics = extract_metrics_from_log(without_balancer_log)

        # Create aggregated results
        aggregated = {
            "experiment_name": config.name,
            "config": {
                "gamma": gamma,
                "d_model": config.d_model,
                "d_head": config.d_head,
                "n_layers": config.n_layers,
                "shard_size": config.shard_size,
                "causal": config.causal,
                "use_flux": config.use_flux,
                "n_ds_layers": config.n_ds_layers,
                "n_ss_layers": config.n_ss_layers,
                "data_codes": config.data_codes,
                "balancer_config": config.balancer_config,
            },
            "with_balancer": with_balancer_metrics,
            "without_balancer": without_balancer_metrics,
        }

        # Write aggregated log
        agg_log_file = log_dir / "aggregated.log"
        with open(agg_log_file, "w") as f:
            f.write(f"Experiment: {config.name}\n")
            f.write("=" * 50 + "\n\n")

            # Configuration
            f.write("Configuration:\n")
            for key, value in aggregated["config"].items():
                f.write(f"{key}: {value}\n")
            f.write("\n" + "-" * 30 + "\n")

            # With balancer results
            f.write("With Balancer:\n")
            for _key, values in with_balancer_metrics.items():
                for value in values:
                    f.write(f"{value}\n")
            f.write("\n" + "-" * 30 + "\n")

            # Without balancer results
            f.write("Without Balancer:\n")
            for _key, values in without_balancer_metrics.items():
                for value in values:
                    f.write(f"{value}\n")
            f.write("\n")

        print(f"Aggregated log written to: {agg_log_file}")

        # Display the contents of the aggregated log
        print(f"\n{'=' * 60}")
        print("Experiment Results:")
        print(f"{'=' * 60}")
        print(agg_log_file.read_text())

        return aggregated

    def run_experiment(self, config: ExperimentConfig, dry_run: bool = False) -> bool:
        """Generate and run a single experiment."""
        print(f"\n{'=' * 60}")
        print(f"Experiment: {config.name}")
        print(f"{'=' * 60}")
        print(f"Data codes: {config.data_codes}")
        print(f"Balancer config: {config.balancer_config}")
        print(f"Model config: d_model={config.d_model}, d_head={config.d_head}, n_layers={config.n_layers}")
        print(
            f"Flux config: use_flux={config.use_flux}, n_ds_layers={config.n_ds_layers}, n_ss_layers={config.n_ss_layers}"
        )
        log_dir_path = f"{config.log_dir}/{self.run_timestamp}_{config.name}"
        print(f"Log directory: {log_dir_path}")

        # Generate script
        script_path = self.generate_script(config)
        print(f"Generated script: {script_path}")

        if dry_run:
            print("DRY RUN: Script generated but not executed")
            return True

        try:
            # Run the experiment
            print("Starting experiment...")
            subprocess.run([str(script_path)], cwd=Path.cwd(), check=True, capture_output=False)
            print("Experiment completed successfully")
            return True

        except subprocess.CalledProcessError as e:
            print(f"Experiment failed with return code {e.returncode}")
            return False

    def run_experiments(self, configs: list[ExperimentConfig], dry_run: bool = False) -> None:
        """Run multiple experiments sequentially with gaps between them."""
        print(f"Running {len(configs)} experiments with {self.gap_seconds}s gaps between them")

        success_count = 0
        all_results = []

        for i, config in enumerate(configs):
            success = self.run_experiment(config, dry_run)
            if success:
                success_count += 1
                if not dry_run:
                    # Store results for final summary
                    try:
                        result = self.aggregate_logs(config)
                        all_results.append(result)
                    except Exception as e:
                        print(f"Warning: Failed to aggregate logs for {config.name}: {e}")

            # Add gap between experiments (except after the last one)
            if i < len(configs) - 1:
                if not dry_run:
                    print(f"\nWaiting {self.gap_seconds} seconds before next experiment...")
                    time.sleep(self.gap_seconds)
                else:
                    print(f"\nDRY RUN: Would wait {self.gap_seconds} seconds before next experiment")

        print(f"\n{'=' * 60}")
        print(f"Experiment Summary: {success_count}/{len(configs)} successful")
        print(f"{'=' * 60}")

        # Generate comprehensive summary if we have results
        if all_results and not dry_run:
            self.generate_comprehensive_summary(all_results, configs[0].log_dir)

    def generate_comprehensive_summary(self, results: list[dict], base_log_dir: str):
        """Generate a comprehensive summary across all experiments."""
        # Add timestamp to make summary unique for each run
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        summary_file = Path(base_log_dir) / f"comprehensive_summary_{timestamp}.log"
        summary_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating comprehensive summary: {summary_file}")

        with open(summary_file, "w") as f:
            f.write("Comprehensive Experiment Summary\n")
            f.write("=" * 50 + "\n\n")

            # Summary table header
            f.write(
                f"{'Experiment':<40} {'Balancer Config':<15} {'With Bal Throughput':<20} {'Without Bal Throughput':<20} {'Improvement':<12}\n"
            )
            f.write("-" * 120 + "\n")

            for result in results:
                name = result["experiment_name"]
                bal_config = result["config"]["balancer_config"]

                # Extract throughput values
                with_throughput = result["with_balancer"].get("Throughput on", [])
                without_throughput = result["without_balancer"].get("Throughput on", [])

                with_val = with_throughput[0] if with_throughput else "N/A"
                without_val = without_throughput[0] if without_throughput else "N/A"

                # Calculate improvement if possible
                improvement = "N/A"
                if with_throughput and without_throughput:
                    try:
                        # Extract numeric values (assuming format like "Throughput on rank 0: 1234.56 tokens/s")
                        with_num = float(re.search(r"([\d.]+)", with_val).group(1))
                        without_num = float(re.search(r"([\d.]+)", without_val).group(1))
                        if without_num > 0:
                            improvement = f"{((with_num - without_num) / without_num * 100):.1f}%"
                    except (ValueError, AttributeError, TypeError):
                        improvement = "N/A"

                f.write(f"{name:<40} {bal_config:<15} {with_val:<20} {without_val:<20} {improvement:<12}\n")

            f.write("\n" + "=" * 50 + "\n")
            f.write("Detailed Results:\n\n")

            # Detailed results for each experiment
            for result in results:
                f.write(f"Experiment: {result['experiment_name']}\n")
                f.write("-" * 30 + "\n")

                # Config
                f.write("Configuration:\n")
                for key, value in result["config"].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

                # Key metrics comparison
                f.write("Performance Comparison:\n")

                # Compare key metrics
                metrics_to_compare = ["Throughput on", "HFU", "Per step time"]
                for metric in metrics_to_compare:
                    with_vals = result["with_balancer"].get(metric, [])
                    without_vals = result["without_balancer"].get(metric, [])

                    if with_vals or without_vals:
                        f.write(f"  {metric}:\n")
                        f.write(f"    With balancer: {with_vals[0] if with_vals else 'N/A'}\n")
                        f.write(f"    Without balancer: {without_vals[0] if without_vals else 'N/A'}\n")

                f.write("\n" + "=" * 30 + "\n\n")

        print(f"Comprehensive summary written to: {summary_file}")

        # Display quick overview
        print("\nQuick Overview:")
        print(f"{'Experiment':<30} {'Config':<10} {'Improvement':<12}")
        print("-" * 55)

        for result in results:
            name = result["experiment_name"][:28]  # Truncate long names
            bal_config = result["config"]["balancer_config"]

            # Calculate improvement
            with_throughput = result["with_balancer"].get("Throughput on", [])
            without_throughput = result["without_balancer"].get("Throughput on", [])

            improvement = "N/A"
            if with_throughput and without_throughput:
                try:
                    with_num = float(re.findall(r"([\d.]+)", with_throughput[0])[1])
                    without_num = float(re.findall(r"([\d.]+)", without_throughput[0])[1])
                    if without_num > 0:
                        improvement = f"{((with_num - without_num) / without_num * 100):+.1f}%"
                except (ValueError, AttributeError, TypeError):
                    improvement = "N/A"

            print(f"{name:<30} {bal_config:<10} {improvement:<12}")

        print("-" * 55)


def get_flux_experiments() -> list[ExperimentConfig]:
    """Get the Flux-based experiment configurations from the original paper script."""
    experiments = []

    # Flux low-res image pretraining experiments
    for balancer_config in ["g1n32", "g2n16", "g4n8", "g8n4"]:
        experiments.append(
            ExperimentConfig(
                name=f"flux_lowres_image_pretraining_{balancer_config}",
                data_codes="g32b32i256f1s0",
                balancer_config=balancer_config,
                d_model=3072,
                d_head=128,
                n_ds_layers=19,
                n_ss_layers=38,
                shard_size=4,
                use_flux=1,
            )
        )

    # Flux mixed-res image pretraining experiments
    for balancer_config in ["g1n32", "g2n16", "g4n8", "g8n4"]:
        experiments.append(
            ExperimentConfig(
                name=f"flux_mixres_image_pretraining_{balancer_config}",
                data_codes="g16b4i256f1s0,g4b5i512f1s0,g4b5i1024f1s0,g8b1i2048f1s0",
                balancer_config=balancer_config,
                d_model=3072,
                d_head=128,
                n_ds_layers=19,
                n_ss_layers=38,
                shard_size=4,
                use_flux=1,
            )
        )

    # Flux joint image/video pretraining experiments
    for balancer_config in ["g1n32", "g2n16", "g4n8", "g8n4"]:
        experiments.append(
            ExperimentConfig(
                name=f"flux_joint_image_video_pretraining_{balancer_config}",
                data_codes="g8b4i256f1s0,g2b5i512f1s0,g2b5i1024f1s0,g4b1i2048f1s0,g1b10i256f4s0,g3b1i512f4s0,g8b2i256f85s1,g4b1i512f85s1",
                balancer_config=balancer_config,
                d_model=3072,
                d_head=128,
                n_ds_layers=19,
                n_ss_layers=38,
                shard_size=4,
                use_flux=1,
            )
        )

    return experiments


def get_default_experiment() -> ExperimentConfig:
    """Get the default experiment configuration from the original run_all.sh script."""
    return ExperimentConfig(
        name="default_experiment",
        data_codes="g8b32i256f1s0",
        balancer_config="g1n8",
        d_model=3072,
        d_head=128,
        n_layers=57,
        shard_size=8,
        causal=0,
        use_flux=0,
        n_ds_layers=19,
        n_ss_layers=38,
    )


def main():
    parser = argparse.ArgumentParser(description="Python-based experiment launcher")
    parser.add_argument(
        "--experiment-type", choices=["default", "flux", "custom"], default="default", help="Type of experiments to run"
    )
    parser.add_argument("--script-dir", default="tmp/scripts", help="Directory to store generated scripts")
    parser.add_argument("--gap-seconds", type=int, default=10, help="Seconds to wait between experiments")
    parser.add_argument("--dry-run", action="store_true", help="Generate scripts but don't run them")
    parser.add_argument("--log-dir", default="logs", help="Base directory for experiment logs")

    # Custom experiment options
    parser.add_argument("--name", help="Custom experiment name")
    parser.add_argument("--data-codes", help="Custom data codes")
    parser.add_argument("--balancer-config", help="Custom balancer configuration")
    parser.add_argument("--d-model", type=int, default=3072, help="Model dimension")
    parser.add_argument("--d-head", type=int, default=128, help="Head dimension")
    parser.add_argument("--n-layers", type=int, default=57, help="Number of layers")
    parser.add_argument("--use-flux", type=int, default=0, help="Use Flux model")

    args = parser.parse_args()

    launcher = ExperimentLauncher(args.script_dir, args.gap_seconds)

    if args.experiment_type == "default":
        config = get_default_experiment()
        config.log_dir = args.log_dir
        launcher.run_experiments([config], args.dry_run)

    elif args.experiment_type == "flux":
        configs = get_flux_experiments()
        for config in configs:
            config.log_dir = args.log_dir
        launcher.run_experiments(configs, args.dry_run)

    elif args.experiment_type == "custom":
        if not args.name or not args.data_codes or not args.balancer_config:
            print("Error: Custom experiments require --name, --data-codes, and --balancer-config")
            return 1

        config = ExperimentConfig(
            name=args.name,
            data_codes=args.data_codes,
            balancer_config=args.balancer_config,
            d_model=args.d_model,
            d_head=args.d_head,
            n_layers=args.n_layers,
            use_flux=args.use_flux,
            log_dir=args.log_dir,
        )
        launcher.run_experiments([config], args.dry_run)

    return 0


if __name__ == "__main__":
    exit(main())
