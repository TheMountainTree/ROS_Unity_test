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
    if len(set(shapes)) == 1:
        return np.stack(epochs, axis=0)

    n_channels_set = set(s[0] for s in shapes)
    if len(n_channels_set) != 1:
        raise ValueError(f"Inconsistent channel counts: {n_channels_set}")

    min_samples = min(s[1] for s in shapes)
    max_samples = max(s[1] for s in shapes)
    print(
        f"Epochs have varying sample counts ({min_samples}-{max_samples}), "
        f"truncating to {min_samples}"
    )
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
    if x.dtype == object:
        print("Converting object array to 3D numeric array...")
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
        dist = ", ".join(f"{int(lbl)}:{int(cnt)}" for lbl, cnt in zip(labels, counts))
        print(f"label distribution: {dist}")
    if x.size > 0 and np.issubdtype(x.dtype, np.number):
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
    candidates = sorted(search_dir.glob("ssvep4_decode_dataset_*.npy"))
    if not candidates:
        raise FileNotFoundError(f"No decode dataset files in: {search_dir}")
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

    print("Dominant frequency by channel:")
    for i in range(n_channels):
        f0, a0 = _dominant_freq(flat[i], fs)
        print(f"  Ch{i+1}: f_peak={f0:.2f} Hz, amp={a0:.3f}")


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

    print(f"Plotting {n_plot} decode epoch(s) to: {out_dir}")
    for ep in range(n_plot):
        n_channels = x.shape[1]
        fig, axes = plt.subplots(n_channels, 1, figsize=(12, 2 * n_channels), sharex=True)
        fig.suptitle(f"Decode Epoch {ep} | label={int(y[ep])}", fontsize=14)

        n_samples = x.shape[2]
        t = np.arange(n_samples) / sample_rate if sample_rate > 0 else np.arange(n_samples)
        for ch in range(n_channels):
            ax = axes[ch] if n_channels > 1 else axes
            ax.plot(t, x[ep, ch, :], linewidth=0.8)
            ax.set_ylabel(f"Ch{ch+1}")
            ax.grid(alpha=0.25, linestyle="--")

        x_label = "Time (s)" if sample_rate > 0 else "Sample"
        (axes[-1] if n_channels > 1 else axes).set_xlabel(x_label)
        fig.tight_layout(rect=[0, 0.02, 1, 0.98])
        save_path = out_dir / f"decode_epoch_{ep:03d}_label_{int(y[ep])}.png"
        fig.savefig(save_path, dpi=120)
        if show:
            plt.show(block=False)
            plt.pause(0.2)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and visualize SSVEP4 decode dataset npy"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Path to decode dataset npy. If empty, use latest ssvep4 decode dataset.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/central_controller_ssvep3/plots_decode",
        help="Directory to save decode epoch plot png files.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=1,
        help="Max decode epochs to plot (default: 1).",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=1000.0,
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
        help="Print additional diagnostic stats.",
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

    can_plot = x.ndim == 3 and x.shape[0] > 0 and y.ndim == 1 and x.shape[0] == y.shape[0]
    if can_plot:
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
