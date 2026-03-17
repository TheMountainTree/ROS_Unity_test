#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def _convert_to_3d(x: np.ndarray) -> np.ndarray:
    """Convert object array of 2D epochs to proper 3D numeric array.

    If epochs have different sample counts, truncates to minimum length.
    """
    if x.ndim != 1 or x.dtype != object:
        return x

    epochs = [np.asarray(epoch, dtype=np.float64) for epoch in x]
    shapes = [e.shape for e in epochs]

    # Check if all shapes are the same
    if len(set(shapes)) == 1:
        return np.stack(epochs, axis=0)

    # Different shapes - truncate to minimum length
    n_channels_set = set(s[0] for s in shapes)
    if len(n_channels_set) != 1:
        raise ValueError(f"Inconsistent channel counts: {n_channels_set}")

    n_channels = shapes[0][0]
    min_samples = min(s[1] for s in shapes)
    max_samples = max(s[1] for s in shapes)
    print(f"Epochs have varying sample counts ({min_samples}-{max_samples}), truncating to {min_samples}")

    truncated = [e[:, :min_samples] for e in epochs]
    return np.stack(truncated, axis=0)


def _load_dataset(path: Path) -> Dict[str, np.ndarray]:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        payload = obj.item()
    elif isinstance(obj, dict):
        payload = obj
    else:
        raise ValueError(f"Unsupported npy payload type: {type(obj)}")

    if not isinstance(payload, dict):
        raise ValueError("Payload is not a dict like {'x': ..., 'y': ...}")
    if "x" not in payload or "y" not in payload:
        raise ValueError("Payload missing required keys 'x' or 'y'")

    x = np.asarray(payload["x"])
    y = np.asarray(payload["y"])

    # Auto-convert object array to 3D numeric array
    if x.dtype == object:
        print(f"Converting object array to 3D numeric array...")
        x = _convert_to_3d(x)

    return {"x": x, "y": y}


def _check_dataset(x: np.ndarray, y: np.ndarray) -> Tuple[bool, str]:
    problems = []

    if x.ndim != 3:
        problems.append(f"x.ndim should be 3, got {x.ndim}")
    if y.ndim != 1:
        problems.append(f"y.ndim should be 1, got {y.ndim}")
    if x.ndim == 3 and y.ndim == 1 and x.shape[0] != y.shape[0]:
        problems.append(f"epoch count mismatch: x={x.shape[0]}, y={y.shape[0]}")
    if x.ndim == 3 and x.shape[1] != 8:
        problems.append(f"channel count expected 8, got {x.shape[1]}")
    if x.size == 0 or y.size == 0:
        problems.append("dataset is empty")
    # Only check finite for numeric dtypes
    if x.size > 0 and np.issubdtype(x.dtype, np.number):
        if not np.isfinite(x).all():
            problems.append("x contains NaN or Inf")
    elif x.size > 0:
        problems.append(f"x has non-numeric dtype: {x.dtype}")
    if y.size > 0 and np.issubdtype(y.dtype, np.number):
        if not np.isfinite(y).all():
            problems.append("y contains NaN or Inf")
    elif y.size > 0:
        problems.append(f"y has non-numeric dtype: {y.dtype}")

    ok = len(problems) == 0
    msg = "OK" if ok else " | ".join(problems)
    return ok, msg


def _print_summary(x: np.ndarray, y: np.ndarray) -> None:
    print("=== Dataset Summary ===")
    print(f"x.shape: {x.shape}, dtype={x.dtype}")
    print(f"y.shape: {y.shape}, dtype={y.dtype}")
    if y.size > 0:
        labels, counts = np.unique(y, return_counts=True)
        dist = ", ".join([f"{int(lbl)}:{int(cnt)}" for lbl, cnt in zip(labels, counts)])
        print(f"label distribution: {dist}")

    # Handle object arrays separately
    if x.dtype == object and x.ndim == 1:
        shapes = [e.shape if hasattr(e, 'shape') else 'N/A' for e in x]
        unique_shapes = set(shapes)
        if len(unique_shapes) == 1:
            print(f"Each epoch shape: {shapes[0]}")
        else:
            print(f"Epoch shapes vary: {unique_shapes}")
            print(f"  min samples: {min(s[1] if len(s) > 1 else 0 for s in shapes if isinstance(s, tuple))}")
            print(f"  max samples: {max(s[1] if len(s) > 1 else 0 for s in shapes if isinstance(s, tuple))}")
        # Compute stats across all epochs
        all_data = np.concatenate([np.asarray(e).flatten() for e in x])
        print(f"x min/max: {all_data.min():.6f} / {all_data.max():.6f}")
        print(f"x mean/std: {all_data.mean():.6f} / {all_data.std():.6f}")
    elif x.size > 0 and np.issubdtype(x.dtype, np.number):
        print(f"x min/max: {x.min():.6f} / {x.max():.6f}")
        print(f"x mean/std: {x.mean():.6f} / {x.std():.6f}")


def _resolve_input_file(input_path: str) -> Path:
    if input_path:
        p = Path(input_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        return p

    search_dir = (Path.cwd() / "data" / "central_controller_ssvep3").resolve()
    if not search_dir.exists():
        raise FileNotFoundError(
            f"No input path provided and default dataset dir not found: {search_dir}"
        )
    candidates = sorted(search_dir.glob("ssvep3_pretrain_dataset_*.npy"))
    if not candidates:
        raise FileNotFoundError(f"No dataset files in: {search_dir}")
    return candidates[-1]


def _dominant_freq(signal: np.ndarray, fs: float) -> Tuple[float, float]:
    x = np.asarray(signal, dtype=float)
    if x.size < 8 or fs <= 0:
        return 0.0, 0.0
    x = x - np.mean(x)
    spec = np.fft.rfft(x)
    amp = np.abs(spec)
    if amp.size <= 1:
        return 0.0, 0.0
    amp[0] = 0.0
    idx = int(np.argmax(amp))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    return float(freqs[idx]), float(amp[idx])


def _run_diagnostic(x: np.ndarray, fs: float) -> None:
    print("=== Diagnostic Mode ===")
    if x.ndim != 3 or x.shape[0] == 0:
        print("Skip diagnostic: x is not valid 3D epochs.")
        return

    n_epochs, n_channels, n_samples = x.shape
    flat = np.transpose(x, (1, 0, 2)).reshape(n_channels, n_epochs * n_samples)

    ch_mean = flat.mean(axis=1)
    ch_std = flat.std(axis=1)
    ch_min = flat.min(axis=1)
    ch_max = flat.max(axis=1)
    ch_ptp = ch_max - ch_min
    zero_ratio = np.mean(np.isclose(flat, 0.0, atol=1e-8), axis=1)

    print("Channel stats (mean/std/min/max/ptp/zero_ratio):")
    for i in range(n_channels):
        print(
            f"  Ch{i+1}: mean={ch_mean[i]:.3f}, std={ch_std[i]:.3f}, "
            f"min={ch_min[i]:.3f}, max={ch_max[i]:.3f}, ptp={ch_ptp[i]:.3f}, "
            f"zero={zero_ratio[i]:.3f}"
        )

    corr = np.corrcoef(flat)
    corr = np.nan_to_num(corr, nan=0.0)
    off_diag = corr[~np.eye(n_channels, dtype=bool)]
    max_abs_corr = float(np.max(np.abs(off_diag))) if off_diag.size > 0 else 0.0
    strong_pairs = int(np.sum(np.abs(off_diag) > 0.98))
    print(f"Max |off-diagonal correlation|: {max_abs_corr:.4f}")
    print(f"Pairs with |corr| > 0.98: {strong_pairs}")

    print("Dominant frequency by channel:")
    for i in range(n_channels):
        f0, a0 = _dominant_freq(flat[i], fs)
        print(f"  Ch{i+1}: f_peak={f0:.2f} Hz, amp={a0:.3f}")

    suspicious = []
    near_zero_std = np.where(ch_std < 1e-6)[0]
    if near_zero_std.size > 0:
        suspicious.append(
            "near-constant channels: "
            + ", ".join([f"Ch{int(i)+1}" for i in near_zero_std])
        )
    mostly_zero = np.where(zero_ratio > 0.95)[0]
    if mostly_zero.size > 0:
        suspicious.append(
            "mostly-zero channels: "
            + ", ".join([f"Ch{int(i)+1}" for i in mostly_zero])
        )
    large_dc = np.where(np.abs(ch_mean) > 5.0 * (ch_std + 1e-9))[0]
    if large_dc.size > 0:
        suspicious.append(
            "strong DC offset channels: "
            + ", ".join([f"Ch{int(i)+1}" for i in large_dc])
        )
    if strong_pairs >= max(2, n_channels):
        suspicious.append("many channels are highly correlated (possible duplicated/test signal)")

    if suspicious:
        print("Suspicious patterns:")
        for item in suspicious:
            print(f"  - {item}")
    else:
        print("No obvious suspicious pattern by current heuristics.")


def _plot_epochs(
    x: np.ndarray,
    y: np.ndarray,
    out_dir: Path,
    max_epochs: int,
    show: bool,
    sample_rate: float,
) -> None:
    if not show:
        import matplotlib
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    n_epochs = x.shape[0]
    n_plot = min(n_epochs, max_epochs) if max_epochs > 0 else n_epochs
    if n_plot == 0:
        print("No epochs to plot.")
        return

    print(f"Plotting {n_plot} epoch(s) to: {out_dir}")

    # Handle object arrays
    if x.dtype == object:
        for ep in range(n_plot):
            epoch_data = np.asarray(x[ep])
            n_channels, n_samples = epoch_data.shape
            fig, axes = plt.subplots(n_channels, 1, figsize=(12, 2 * n_channels), sharex=True)
            fig.suptitle(f"Epoch {ep} | label={int(y[ep])}", fontsize=14)

            t = np.arange(n_samples) / sample_rate if sample_rate > 0 else np.arange(n_samples)
            for ch in range(n_channels):
                ax = axes[ch] if n_channels > 1 else axes
                ax.plot(t, epoch_data[ch, :], linewidth=0.8)
                ax.set_ylabel(f"Ch{ch+1}")
                ax.grid(alpha=0.25, linestyle="--")

            (axes[-1] if n_channels > 1 else axes).set_xlabel("Time (s)" if sample_rate > 0 else "Sample")
            fig.tight_layout(rect=[0, 0.02, 1, 0.98])
            save_path = out_dir / f"epoch_{ep:03d}_label_{int(y[ep])}.png"
            fig.savefig(save_path, dpi=120)
            if show:
                plt.show(block=False)
                plt.pause(0.2)
            plt.close(fig)
        return

    # Handle 3D numeric arrays
    for ep in range(n_plot):
        fig, axes = plt.subplots(8, 1, figsize=(12, 14), sharex=True)
        fig.suptitle(f"Epoch {ep} | label={int(y[ep])}", fontsize=14)

        n_samples = x.shape[2]
        t = np.arange(n_samples) / sample_rate if sample_rate > 0 else np.arange(n_samples)
        for ch in range(8):
            ax = axes[ch]
            ax.plot(t, x[ep, ch, :], linewidth=0.8)
            ax.set_ylabel(f"Ch{ch+1}")
            ax.grid(alpha=0.25, linestyle="--")

        axes[-1].set_xlabel("Time (s)" if sample_rate > 0 else "Sample")
        fig.tight_layout(rect=[0, 0.02, 1, 0.98])
        save_path = out_dir / f"epoch_{ep:03d}_label_{int(y[ep])}.png"
        fig.savefig(save_path, dpi=120)
        if show:
            plt.show(block=False)
            plt.pause(0.2)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and visualize SSVEP3 pretrain dataset npy"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Path to dataset npy. If empty, use latest in data/central_controller_ssvep3.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/central_controller_ssvep3/plots",
        help="Directory to save epoch plot png files.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=0,
        help="Max epochs to plot (0 = all).",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=250.0,
        help="Sampling rate for x-axis in visualization.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show matplotlib windows while saving plots.",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Print additional diagnostic stats (channel stats/correlation/frequency).",
    )
    args = parser.parse_args()

    input_path = _resolve_input_file(args.input)
    print(f"Using dataset: {input_path}")
    payload = _load_dataset(input_path)
    x = payload["x"]
    y = payload["y"]

    ok, msg = _check_dataset(x, y)
    print(f"Validation: {msg}")
    _print_summary(x, y)
    if args.diagnostic and x.ndim == 3:
        _run_diagnostic(x, args.sample_rate)

    # Plot if we have valid epochs (3D array or object array of epochs)
    can_plot = (x.ndim == 3 and x.shape[0] > 0) or (x.dtype == object and x.shape[0] > 0)
    if can_plot and y.ndim == 1 and x.shape[0] == y.shape[0]:
        out_dir = Path(args.out_dir).expanduser().resolve()
        _plot_epochs(
            x=x,
            y=y,
            out_dir=out_dir,
            max_epochs=args.max_epochs,
            show=args.show,
            sample_rate=args.sample_rate,
        )
    else:
        print("Skip plotting due to invalid or empty dataset.")

    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
