"""
Skin Tone Fairness Evaluation Script for rPPG Toolbox.

Trains baseline vs physics-augmented models on MMPD and evaluates
per-Fitzpatrick-skin-tone metrics to measure fairness improvement.

Usage:
    python tools/run_skin_tone_experiment.py \
        --base_config configs/train_configs/MMPD_MMPD_MMPD_TSCAN_BASIC.yaml \
        --output_dir experiments/skin_tone_aug \
        --epochs 30
"""

import argparse
import copy
import csv
import os
import subprocess
import sys
from pathlib import Path

import yaml

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset.data_manager import ensure_dataset, get_data_dir

SKIN_TONES = [3, 4, 5, 6]


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def generate_train_config(base_cfg, output_dir, variant, epochs=None):
    """Generate a training config for baseline or augmented variant."""
    cfg = copy.deepcopy(base_cfg)

    if epochs is not None:
        cfg["TRAIN"]["EPOCHS"] = epochs

    model_name = cfg["TRAIN"].get("MODEL_FILE_NAME", "model")
    cfg["TRAIN"]["MODEL_FILE_NAME"] = f"{model_name}_{variant}"
    cfg["LOG"] = {"PATH": os.path.join(output_dir, "runs", variant)}

    # Only set SKIN_COLOR on splits that use MMPD (which has Fitzpatrick labels).
    # UBFC-rPPG and other datasets don't have INFO filtering, so leave them alone.
    for split in ["TRAIN", "VALID", "TEST"]:
        if split in cfg:
            dataset = cfg[split].get("DATA", {}).get("DATASET", "")
            if dataset == "MMPD":
                cfg[split]["DATA"].setdefault("INFO", {})["SKIN_COLOR"] = SKIN_TONES

    if variant == "augmented":
        cfg["TRAIN"]["DATA"].setdefault("AUGMENTATION", {})["PHYSICS_SKIN_TONE"] = True
        cfg["TRAIN"]["DATA"]["AUGMENTATION"]["PHYSICS_SKIN_P"] = 0.5
        # Augmentation needs Raw data type to be present
        data_types = cfg["TRAIN"]["DATA"].get("PREPROCESS", {}).get("DATA_TYPE", [])
        if "Raw" not in data_types:
            data_types.append("Raw")
            cfg["TRAIN"]["DATA"]["PREPROCESS"]["DATA_TYPE"] = data_types
    else:
        cfg["TRAIN"]["DATA"].setdefault("AUGMENTATION", {})["PHYSICS_SKIN_TONE"] = False

    config_path = os.path.join(output_dir, "configs", f"train_{variant}.yaml")
    save_yaml(cfg, config_path)
    return config_path


def generate_test_config(base_cfg, output_dir, variant, skin_tone, model_path):
    """Generate an only_test config for a specific skin tone."""
    cfg = copy.deepcopy(base_cfg)
    cfg["TOOLBOX_MODE"] = "only_test"

    # Remove TRAIN and VALID sections — not needed for only_test
    cfg.pop("TRAIN", None)
    cfg.pop("VALID", None)

    if skin_tone == "all":
        cfg["TEST"]["DATA"]["INFO"]["SKIN_COLOR"] = SKIN_TONES
        label = "all"
    else:
        cfg["TEST"]["DATA"]["INFO"]["SKIN_COLOR"] = [skin_tone]
        label = f"skin{skin_tone}"

    cfg["INFERENCE"]["MODEL_PATH"] = model_path
    cfg["LOG"] = {"PATH": os.path.join(output_dir, "runs", f"{variant}_test_{label}")}

    # Ensure test dataset is MMPD for per-skin-tone eval
    cfg["TEST"]["DATA"]["DATASET"] = "MMPD"

    config_path = os.path.join(output_dir, "configs", f"test_{variant}_{label}.yaml")
    save_yaml(cfg, config_path)
    return config_path


def find_best_model(log_dir):
    """Find the best (or last) saved model checkpoint in a training run."""
    model_dir = os.path.join(log_dir, "PreTrainedModels")
    if not os.path.isdir(model_dir):
        # Search recursively under log_dir
        for root, dirs, files in os.walk(log_dir):
            for f in files:
                if f.endswith(".pth"):
                    return os.path.join(root, f)
        return None

    pth_files = sorted(Path(model_dir).rglob("*.pth"))
    if not pth_files:
        return None
    # Prefer files with "best" in name, else take last
    best = [p for p in pth_files if "best" in p.name.lower()]
    return str(best[0]) if best else str(pth_files[-1])


def run_main(config_path):
    """Run main.py with a given config and return the subprocess result."""
    cmd = [sys.executable, "main.py", "--config_file", config_path]
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"WARNING: main.py exited with code {result.returncode} for {config_path}")
    return result


def run_test_and_collect(config_path):
    """Run main.py in only_test mode and parse metrics from stdout."""
    cmd = [sys.executable, "main.py", "--config_file", config_path]
    print(f"\n{'='*60}")
    print(f"Testing: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    # Parse metrics from stdout
    metrics = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        for metric_name in ["MAE", "RMSE", "MAPE", "Pearson", "SNR", "MACC"]:
            if metric_name in line and "+/-" in line:
                try:
                    # Format: "FFT MAE (FFT Label): 3.45 +/- 0.12"
                    value_part = line.split(":")[1].strip()
                    value = float(value_part.split("+/-")[0].strip())
                    # Use last matching metric name to handle both FFT/Peak prefixes
                    metrics[metric_name] = value
                except (IndexError, ValueError):
                    pass
    return metrics


def generate_report(all_results, output_dir):
    """Generate comparison report as CSV and printed table."""
    os.makedirs(output_dir, exist_ok=True)

    # CSV output
    csv_path = os.path.join(output_dir, "skin_tone_comparison.csv")
    metric_names = ["MAE", "RMSE", "MAPE", "Pearson", "SNR", "MACC"]

    rows = []
    for variant in ["baseline", "augmented"]:
        for label, metrics in sorted(all_results.get(variant, {}).items()):
            row = {"variant": variant, "skin_tone": label}
            for m in metric_names:
                row[m] = metrics.get(m, "")
            rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "skin_tone"] + metric_names)
        writer.writeheader()
        writer.writerows(rows)

    # Print summary table
    print(f"\n{'='*80}")
    print("SKIN TONE FAIRNESS COMPARISON")
    print(f"{'='*80}")
    header = f"{'Variant':<12} {'Skin Tone':<12} {'MAE':>8} {'RMSE':>8} {'MAPE':>8} {'Pearson':>8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['variant']:<12} {row['skin_tone']:<12} "
            f"{row.get('MAE', 'N/A'):>8.3f} {row.get('RMSE', 'N/A'):>8.3f} "
            f"{row.get('MAPE', 'N/A'):>8.3f} {row.get('Pearson', 'N/A'):>8.3f}"
            if isinstance(row.get("MAE"), (int, float))
            else f"{row['variant']:<12} {row['skin_tone']:<12} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}"
        )

    # Fairness metrics (disparity across skin tones)
    print(f"\n{'='*80}")
    print("FAIRNESS METRICS (per-skin-tone disparity)")
    print(f"{'='*80}")
    for variant in ["baseline", "augmented"]:
        variant_data = all_results.get(variant, {})
        # Only per-tone entries (exclude 'all')
        tone_metrics = {k: v for k, v in variant_data.items() if k != "all"}
        if not tone_metrics:
            continue
        print(f"\n  {variant.upper()}:")
        for metric in ["MAE", "RMSE", "MAPE"]:
            values = [v[metric] for v in tone_metrics.values() if metric in v and isinstance(v[metric], (int, float))]
            if len(values) >= 2:
                import numpy as np
                disparity = max(values) - min(values)
                std = np.std(values)
                print(f"    {metric}: max_disparity={disparity:.3f}, std={std:.3f}, "
                      f"range=[{min(values):.3f}, {max(values):.3f}]")

    print(f"\nResults saved to: {csv_path}")

    # Optional bar chart
    try:
        generate_chart(all_results, output_dir)
    except ImportError:
        print("matplotlib not available, skipping chart generation.")


def generate_chart(all_results, output_dir):
    """Generate grouped bar chart comparing baseline vs augmented per skin tone."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    metric = "MAE"  # Primary fairness metric
    tones = [f"skin{t}" for t in SKIN_TONES]
    x = np.arange(len(tones))
    width = 0.35

    baseline_vals = []
    augmented_vals = []
    for tone in tones:
        b = all_results.get("baseline", {}).get(tone, {}).get(metric, 0)
        a = all_results.get("augmented", {}).get(tone, {}).get(metric, 0)
        baseline_vals.append(b)
        augmented_vals.append(a)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, baseline_vals, width, label="Baseline", color="#4a90d9")
    ax.bar(x + width / 2, augmented_vals, width, label="Augmented", color="#e67e22")

    ax.set_xlabel("Fitzpatrick Skin Type")
    ax.set_ylabel(f"{metric}")
    ax.set_title(f"{metric} by Skin Tone: Baseline vs Physics-Augmented")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Type {t}" for t in SKIN_TONES])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    chart_path = os.path.join(output_dir, "skin_tone_comparison.png")
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved to: {chart_path}")


def main():
    parser = argparse.ArgumentParser(description="Skin Tone Fairness Evaluation for rPPG")
    parser.add_argument("--base_config", required=True, help="Path to base MMPD YAML config")
    parser.add_argument("--output_dir", default="experiments/skin_tone_aug", help="Output directory")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--skip_train", action="store_true", help="Skip training, use existing models")
    parser.add_argument("--baseline_model", default=None, help="Path to pre-trained baseline model (.pth)")
    parser.add_argument("--augmented_model", default=None, help="Path to pre-trained augmented model (.pth)")
    parser.add_argument("--test_data_path", default=None, help="Override TEST.DATA.DATA_PATH")
    parser.add_argument("--test_cached_path", default=None, help="Override TEST.DATA.CACHED_PATH")
    parser.add_argument("--auto_download", action="store_true",
                        help="Auto-download datasets and resolve paths via data manager")
    parser.add_argument("--data_dir", default=None,
                        help="Override data directory (alternative to RPPG_DATA_DIR env var)")
    args = parser.parse_args()

    base_cfg = load_yaml(args.base_config)
    output_dir = os.path.abspath(args.output_dir)

    # Auto-download datasets and patch config paths
    if args.auto_download:
        data_dir = get_data_dir(args.data_dir)
        print(f"[auto_download] Data directory: {data_dir}")

        # Ensure MMPD is available (required for skin tone evaluation)
        try:
            mmpd_path = ensure_dataset("MMPD", data_dir=args.data_dir)
        except FileNotFoundError as e:
            print(f"\n{e}", file=sys.stderr)
            sys.exit(1)

        # Patch DATA_PATH in all splits that use MMPD
        for split in ["TRAIN", "VALID", "TEST"]:
            if split in base_cfg:
                ds = base_cfg[split].get("DATA", {}).get("DATASET", "")
                if ds == "MMPD":
                    base_cfg[split]["DATA"]["DATA_PATH"] = mmpd_path
                    cached = os.path.join(mmpd_path, "preprocessed")
                    base_cfg[split]["DATA"]["CACHED_PATH"] = cached

        # If any split uses UBFC-rPPG, ensure it too
        ubfc_needed = any(
            base_cfg.get(s, {}).get("DATA", {}).get("DATASET", "") == "UBFC-rPPG"
            for s in ["TRAIN", "VALID", "TEST"]
        )
        if ubfc_needed:
            try:
                ubfc_path = ensure_dataset("UBFC-rPPG", data_dir=args.data_dir)
            except (FileNotFoundError, ImportError) as e:
                print(f"\n{e}", file=sys.stderr)
                sys.exit(1)
            for split in ["TRAIN", "VALID", "TEST"]:
                if split in base_cfg:
                    ds = base_cfg[split].get("DATA", {}).get("DATASET", "")
                    if ds == "UBFC-rPPG":
                        base_cfg[split]["DATA"]["DATA_PATH"] = ubfc_path
                        cached = os.path.join(ubfc_path, "preprocessed")
                        base_cfg[split]["DATA"]["CACHED_PATH"] = cached

    # Apply test data path overrides to base config (these take priority)
    if args.test_data_path:
        base_cfg.setdefault("TEST", {}).setdefault("DATA", {})["DATA_PATH"] = args.test_data_path
    if args.test_cached_path:
        base_cfg.setdefault("TEST", {}).setdefault("DATA", {})["CACHED_PATH"] = args.test_cached_path

    # Verify TEST section uses MMPD (required for Fitzpatrick skin tone labels)
    test_dataset = base_cfg.get("TEST", {}).get("DATA", {}).get("DATASET", "")
    if test_dataset != "MMPD":
        print(f"WARNING: TEST.DATA.DATASET is '{test_dataset}', expected 'MMPD'. "
              "Overriding to MMPD for per-skin-tone evaluation.")
        base_cfg.setdefault("TEST", {}).setdefault("DATA", {})["DATASET"] = "MMPD"

    model_paths = {}

    # ── Phase 1: Training ──
    if not args.skip_train:
        for variant in ["baseline", "augmented"]:
            config_path = generate_train_config(base_cfg, output_dir, variant, args.epochs)
            print(f"\n{'#'*60}")
            print(f"# TRAINING: {variant.upper()}")
            print(f"# Config: {config_path}")
            print(f"{'#'*60}")
            run_main(config_path)

            log_dir = os.path.join(output_dir, "runs", variant)
            model_path = find_best_model(log_dir)
            if model_path:
                model_paths[variant] = model_path
                print(f"Found model for {variant}: {model_path}")
            else:
                print(f"ERROR: No model found for {variant} in {log_dir}")
    else:
        if args.baseline_model:
            model_paths["baseline"] = args.baseline_model
        if args.augmented_model:
            model_paths["augmented"] = args.augmented_model

    if not model_paths:
        print("ERROR: No models available for testing. Exiting.")
        sys.exit(1)

    # ── Phase 2: Per-skin-tone evaluation ──
    all_results = {}
    test_labels = [f"skin{t}" for t in SKIN_TONES] + ["all"]

    for variant, model_path in model_paths.items():
        all_results[variant] = {}
        for label in test_labels:
            skin_tone = int(label.replace("skin", "")) if label != "all" else "all"
            config_path = generate_test_config(base_cfg, output_dir, variant, skin_tone, model_path)
            metrics = run_test_and_collect(config_path)
            all_results[variant][label] = metrics
            print(f"  {variant} / {label}: {metrics}")

    # ── Phase 3: Report ──
    generate_report(all_results, output_dir)


if __name__ == "__main__":
    main()
