import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_dataset(npy_path):
    data = np.load(npy_path, allow_pickle=True).item()
    x = data.get("x", np.asarray([], dtype=object))
    y = data.get("y", np.zeros((0,), dtype=np.int32))
    return x, y


def plot_trial_eeg(epoch, trial_id, target_id, fs=1000.0, output_dir=None, show=True):
    n_channels, n_samples = epoch.shape
    time_axis = np.arange(n_samples) / fs

    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 2 * n_channels), sharex=True)
    if n_channels == 1:
        axes = [axes]

    for ch in range(n_channels):
        axes[ch].plot(time_axis, epoch[ch, :], linewidth=0.5)
        axes[ch].set_ylabel(f"Ch {ch + 1}")
        axes[ch].grid(True, linestyle=":", alpha=0.5)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Trial {trial_id} | Target {target_id}", fontsize=12)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(
            output_dir, f"trial_{trial_id:03d}_target_{target_id}.png"
        )
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot 8-channel EEG for each trial from npy dataset"
    )
    parser.add_argument(
        "npy_path", type=str, help="Path to ssvep4_pretrain_dataset_*.npy file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (optional)",
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Do not display plots, only save"
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=1000.0,
        help="Sampling frequency in Hz (default: 1000)",
    )
    parser.add_argument(
        "--trials",
        type=str,
        default=None,
        help="Trial indices to plot, e.g., '0,2,5' or '0-10'",
    )
    args = parser.parse_args()

    npy_path = Path(args.npy_path).resolve()
    if not npy_path.exists():
        print(f"Error: file not found: {npy_path}")
        return

    x, y = load_dataset(str(npy_path))
    n_trials = len(x)
    print(f"Loaded {n_trials} trials from {npy_path}")

    if n_trials == 0:
        print("No trials to plot")
        return

    trial_indices = list(range(n_trials))
    if args.trials:
        indices = set()
        for part in args.trials.split(","):
            part = part.strip()
            if "-" in part:
                start, end = map(int, part.split("-"))
                indices.update(range(start, end + 1))
            else:
                indices.add(int(part))
        trial_indices = sorted([i for i in indices if 0 <= i < n_trials])

    show = not args.no_show
    for idx in trial_indices:
        epoch = x[idx]
        target_id = int(y[idx])
        plot_trial_eeg(
            epoch=epoch,
            trial_id=idx + 1,
            target_id=target_id,
            fs=args.fs,
            output_dir=args.output_dir,
            show=show,
        )


if __name__ == "__main__":
    main()
