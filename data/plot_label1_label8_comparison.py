#!/usr/bin/env python3
"""
绘制 Label 1 和 Label 8 的第一个 trial 对比图。
每个 Label 单独一张图，8 个通道分开显示。
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path


def plot_label_comparison(npy_path: str, output_dir: str):
    """绘制 Label 1 和 Label 8 的第一个 trial 对比图。"""
    # Load data
    data = np.load(npy_path, allow_pickle=True).item()
    x = data["x"]  # Shape: (n_trials,) - object array
    y = data["y"]  # Shape: (n_trials,) - labels

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Channel names (according to data analysis docs)
    channel_names = ["O1", "O2", "Oz", "PO3", "PO4", "Pz", "P3", "P4"]

    # Target labels to plot
    target_labels = [1, 8]

    for label in target_labels:
        # Get indices for this label
        indices = np.where(y == label)[0]

        if len(indices) == 0:
            print(f"Warning: No trials found for Label {label}")
            continue

        # Get the first trial for this label
        first_trial_idx = indices[0]
        trial_data = x[first_trial_idx]  # Shape: (8, n_samples)
        n_channels = trial_data.shape[0]
        n_samples = trial_data.shape[1]

        print(f"\nLabel {label} - First trial (index {first_trial_idx}):")
        print(f"  Shape: {trial_data.shape}")
        print(f"  Data range: [{trial_data.min():.2f}, {trial_data.max():.2f}]")

        # Create figure with 8 subplots (one per channel)
        fig, axes = plt.subplots(n_channels, 1, figsize=(14, 16), sharex=True)
        fig.suptitle(
            f"Label {label} - First Trial (Trial Index: {first_trial_idx})\n"
            f"Samples: {n_samples} | Channels: {n_channels}",
            fontsize=14,
            fontweight="bold",
        )

        # Time axis in seconds (assuming 256Hz sampling rate)
        time = np.arange(n_samples) / 256.0

        for ch in range(n_channels):
            ax = axes[ch]
            ch_data = trial_data[ch, :]

            # Plot the channel data
            ax.plot(time, ch_data, color="steelblue", linewidth=0.8)

            # Add zero line for reference
            ax.axhline(y=0, color="red", linestyle="--", alpha=0.3, linewidth=0.5)

            # Channel label
            ax.set_ylabel(
                f"{channel_names[ch]}\n({ch_data.min():.0f} ~ {ch_data.max():.0f})",
                fontsize=9,
            )
            ax.grid(True, alpha=0.3)

            # Set y-axis limits with some padding
            margin = (ch_data.max() - ch_data.min()) * 0.1
            ax.set_ylim(ch_data.min() - margin, ch_data.max() + margin)

        # Common x-axis label
        axes[-1].set_xlabel("Time (seconds)", fontsize=10)

        plt.tight_layout()

        # Save figure
        save_path = output_path / f"label_{label}_first_trial.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
        plt.close(fig)

    print(f"\nAll plots saved to: {output_path.absolute()}")


if __name__ == "__main__":
    npy_file = "data/central_controller_ssvep_node4_test/ssvep4_pretrain_dataset_20260414_161545.npy"
    output_folder = "data/comparison_label1_label8"

    plot_label_comparison(npy_file, output_folder)
