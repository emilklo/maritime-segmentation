#!/usr/bin/env python3
"""Launch RF-DETR-Seg experiments on Idun HPC via SLURM."""

import argparse
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.training.registry import register_experiment

# Resource profiles per model size
RESOURCE_PROFILES = {
    "nano": {"mem": "48G", "time": "8:00:00"},
    "small": {"mem": "48G", "time": "8:00:00"},
    "medium": {"mem": "64G", "time": "12:00:00"},
    "large": {"mem": "64G", "time": "16:00:00"},
    "xlarge": {"mem": "80G", "time": "24:00:00"},
    "2xlarge": {"mem": "80G", "time": "24:00:00"},
}

SLURM_TEMPLATE = textwrap.dedent("""\
    #!/bin/bash
    #==============================================================================
    # SLURM Job: {job_name}
    # Experiment: {experiment_id}
    #==============================================================================

    #SBATCH --job-name={job_name}
    #SBATCH --account={account}
    #SBATCH --output={output_dir}/slurm-%j.out
    #SBATCH --error={output_dir}/slurm-%j.err

    #SBATCH --partition=GPUQ
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=8
    #SBATCH --mem={mem}
    #SBATCH --gres=gpu:a100:1
    #SBATCH --time={time}

    {mail_lines}

    set -euo pipefail

    echo "========================================"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Experiment: {experiment_id}"
    echo "Model size: {model_size}"
    echo "Node: $SLURM_NODELIST"
    echo "Started: $(date)"
    echo "========================================"

    cd "$SLURM_SUBMIT_DIR"

    module purge
    module load Python/3.11.5-GCCcore-13.2.0
    module load CUDA/12.8.0

    echo ""
    echo "=== Environment ==="
    echo "Python: $(which python)"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""

    if ! command -v uv &> /dev/null; then
        echo "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi

    echo "=== Installing Dependencies ==="
    uv sync
    echo ""

    export MASTER_ADDR=localhost
    export CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=0
    export TORCH_CUDNN_V8_API_ENABLED=0
    export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
    export WORLD_SIZE=1
    export RANK=0
    export LOCAL_RANK=0

    echo "=== Starting Training ==="
    uv run python scripts/train.py \\
        --config "{config_path}" \\
        --model-size "{model_size}" \\
        --output-dir "{output_dir}" \\
        --experiment-id "{experiment_id}" \\
        --batch-size {batch_size} \\
        --grad-accum {grad_accum} \\
        --tensorboard

    echo ""
    echo "========================================"
    echo "Finished: $(date)"
    echo "========================================"
""")


def load_experiment_config(path: str | Path) -> dict:
    """Load an experiment config YAML file."""
    with open(path) as f:
        config = yaml.safe_load(f)
    config["_config_path"] = str(path)
    return config


def generate_experiment_id(config: dict) -> str:
    """Generate a unique experiment ID from config name + timestamp."""
    name = config.get("experiment", {}).get("name", "unnamed")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{name}-{timestamp}"


def generate_slurm_script(
    config: dict,
    experiment_id: str,
    output_dir: str,
    account: str = "ie-idi",
    mail_user: str | None = None,
) -> str:
    """Generate a SLURM script from experiment config."""
    exp_cfg = config.get("experiment", {})
    training_cfg = config.get("training", {})

    model_size = exp_cfg.get("model_size", "medium")
    resources = RESOURCE_PROFILES.get(model_size, RESOURCE_PROFILES["medium"])

    batch_size = training_cfg.get("batch_size", 4)
    # Reduce batch size for SLURM (cuDNN disabled = higher memory)
    slurm_batch_size = max(1, batch_size // 2)
    grad_accum = training_cfg.get("grad_accum_steps", 4)
    # Compensate for halved batch size
    slurm_grad_accum = grad_accum * 2

    mail_lines = ""
    if mail_user:
        mail_lines = (
            f"#SBATCH --mail-type=BEGIN,END,FAIL\n"
            f"#SBATCH --mail-user={mail_user}"
        )

    return SLURM_TEMPLATE.format(
        job_name=f"rfdetr-{model_size}-{exp_cfg.get('name', 'exp')}",
        experiment_id=experiment_id,
        account=account,
        output_dir=output_dir,
        mem=resources["mem"],
        time=resources["time"],
        mail_lines=mail_lines,
        model_size=model_size,
        config_path=config.get("_config_path", ""),
        batch_size=slurm_batch_size,
        grad_accum=slurm_grad_accum,
    )


def launch_experiment(
    config_path: str | Path,
    account: str = "ie-idi",
    mail_user: str | None = None,
    dry_run: bool = False,
) -> tuple[str, str | None]:
    """Launch a single experiment.

    Returns:
        Tuple of (experiment_id, slurm_job_id or None for dry_run).
    """
    config = load_experiment_config(config_path)
    experiment_id = generate_experiment_id(config)
    output_dir = f"checkpoints/{experiment_id}"

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate SLURM script
    slurm_script = generate_slurm_script(
        config, experiment_id, output_dir, account=account, mail_user=mail_user,
    )

    slurm_path = Path(output_dir) / "job.slurm"
    slurm_path.write_text(slurm_script)

    if dry_run:
        print(f"[DRY RUN] Experiment: {experiment_id}")
        print(f"  Config: {config_path}")
        print(f"  Model size: {config.get('experiment', {}).get('model_size', '?')}")
        print(f"  Output: {output_dir}")
        print(f"  SLURM script: {slurm_path}")
        print()
        print("--- Generated SLURM script ---")
        print(slurm_script)
        print("--- End SLURM script ---")
        return experiment_id, None

    # Submit job
    result = subprocess.run(
        ["sbatch", str(slurm_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error submitting job: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    # Parse job ID from "Submitted batch job 12345"
    slurm_job_id = result.stdout.strip().split()[-1]

    # Register in experiment registry
    register_experiment(config, slurm_job_id=slurm_job_id, output_dir=output_dir)

    print(f"Launched experiment: {experiment_id}")
    print(f"  SLURM job ID: {slurm_job_id}")
    print(f"  Config: {config_path}")
    print(f"  Model size: {config.get('experiment', {}).get('model_size', '?')}")
    print(f"  Output: {output_dir}")

    return experiment_id, slurm_job_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch RF-DETR-Seg experiments on Idun HPC"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a single experiment config YAML",
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=None,
        help="Paths to multiple experiment config YAMLs",
    )
    parser.add_argument(
        "--account",
        type=str,
        default="ie-idi",
        help="SLURM account (default: ie-idi)",
    )
    parser.add_argument(
        "--mail-user",
        type=str,
        default=None,
        help="Email for SLURM notifications",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate SLURM script but don't submit",
    )
    args = parser.parse_args()

    if not args.config and not args.configs:
        parser.error("Provide --config or --configs")

    config_paths = []
    if args.config:
        config_paths.append(args.config)
    if args.configs:
        config_paths.extend(args.configs)

    print("=" * 60)
    print("RF-DETR-Seg Experiment Launcher")
    print(f"Launching {len(config_paths)} experiment(s)")
    print("=" * 60)
    print()

    results = []
    for path in config_paths:
        exp_id, job_id = launch_experiment(
            path, account=args.account, mail_user=args.mail_user, dry_run=args.dry_run,
        )
        results.append((exp_id, job_id))
        print()

    if not args.dry_run:
        print("=" * 60)
        print("Summary:")
        for exp_id, job_id in results:
            print(f"  {exp_id} -> SLURM job {job_id}")
        print()
        print("Monitor with: uv run python scripts/experiment_status.py")
        print("=" * 60)


if __name__ == "__main__":
    main()
