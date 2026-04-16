#!/usr/bin/env python3
"""
Plot pretrain EEG data by label.
For each label (1-8), plot all trials and channels.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path


def plot_pretrain_by_label(npy_path: str, output_dir: str):
    """Load pretrain npy and plot EEG data grouped by label."""
    # Load data
    data = np.load(npy_path, allow_pickle=True).item()
    x = data["x"]  # Shape: (n_trials,)
    y = data["y"]  # Shape: (n_trials,)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get unique labels (1-8)
    unique_labels = np.unique(y)
    n_channels = x[0].shape[0]  # 8 channels

    print(f"Data info: {len(x)} trials, {n_channels} channels")
    print(f"Labels: {unique_labels}")
    print(f"Timepoints vary per trial (2019-2023 range)")

    # Plot for each label
    for label in unique_labels:
        # Get indices for this label
        indices = np.where(y == label)[0]
        n_trials = len(indices)

        # Create figure with subplots for each channel
        fig, axes = plt.subplots(n_channels, 1, figsize=(14, 12), sharex=False)
        fig.suptitle(
            f"Label {label} - {n_trials} Trials", fontsize=16, fontweight="bold"
        )

        # Colors for different trials
        colors = plt.cm.tab10(np.linspace(0, 1, n_trials))

        for ch in range(n_channels):
            ax = axes[ch]

            # Plot each trial for this channel
            for i, idx in enumerate(indices):
                trial_data = x[idx][ch, :]
                timepoints = np.arange(len(trial_data))
                ax.plot(
                    timepoints,
                    trial_data,
                    color=colors[i],
                    alpha=0.7,
                    linewidth=0.8,
                    label=f"Trial {i + 1}",
                )

            ax.set_ylabel(f"Ch {ch + 1}", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=7, ncol=3)

        axes[-1].set_xlabel("Timepoints", fontsize=10)
        plt.tight_layout()

        # Save figure
        save_path = output_path / f"label_{label}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close(fig)

    print(f"\nAll plots saved to: {output_path.absolute()}")


if __name__ == "__main__":
    npy_file = "data/central_controller_ssvep_node4_test/ssvep4_pretrain_dataset_20260414_161545.npy"
    output_folder = "data/pretrain_plots_20260414_161545"

    plot_pretrain_by_label(npy_file, output_folder)
