#!/usr/bin/env python3
"""
FFT 频谱分析脚本

统一配置区支持:
1. 每个 label 分析 trial 数量或指定 trial 序号
2. 排除坏道（8 通道顺序: O1,O2,Oz,PO3,PO4,Pz,P3,P4）
3. 预处理参数（高通、陷波、重采样等）
4. 峰值搜索参数（频段、阈值、峰间距、top-k）
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import matplotlib
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import butter, detrend, filtfilt, find_peaks, iirnotch, resample, sosfiltfilt

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CHANNEL_NAMES = ["O1", "O2", "Oz", "PO3", "PO4", "Pz", "P3", "P4"]
TARGET_FREQS = [8.684, 9.706, 11.0, 11.786, 12.692, 13.75, 15.0, 18.333]


@dataclass
class PreprocessConfig:
    enabled: bool = False
    highpass_cutoff_hz: float = 6.0
    highpass_order: int = 4
    notch_freqs_hz: List[float] = field(default_factory=lambda: [50.0, 100.0])
    notch_q: float = 35.0
    target_fs: float = 256.0


@dataclass
class PeakDetectConfig:
    min_hz: float = 1.0
    max_hz: float = 60.0
    peak_height_ratio: float = 0.1
    peak_min_distance_bins: int = 5
    top_k: int = 5


@dataclass
class FFTAnalysisConfig:
    npy_path: str = (
        "data/central_controller_ssvep_node4_test/"
        "ssvep4_pretrain_dataset_20260416_164519.npy"
    )
    fs: float = 256.0

    # trial 配置: trial_indices_per_label 有值时优先
    trial_count_per_label: int = 1
    trial_indices_per_label: Optional[List[int]] = None  # 示例: [0, 1, 3]

    # 坏道排除配置: 支持通道名或索引字符串, 例如 ["Oz", "P3"] 或 ["2", "6"]
    exclude_channels: List[str] = field(default_factory=list)

    output_dir_raw: str = "data/fft_analysis"
    output_dir_preprocessed: str = "data/fft_analysis_preprocessed"

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    peak: PeakDetectConfig = field(default_factory=PeakDetectConfig)


# ============================ 统一配置区 ============================
CONFIG = FFTAnalysisConfig(
    trial_count_per_label=5,
    trial_indices_per_label=None,
    exclude_channels=[],#["PO4", "O2", "P4"],
    preprocess=PreprocessConfig(
        enabled=True,
        highpass_cutoff_hz=6.0,
        highpass_order=4,
        notch_freqs_hz=[],#[50.0, 100.0, 12.8, 38.4],
        notch_q=35.0,
        target_fs=256.0,
    ),
    peak=PeakDetectConfig(
        min_hz=1.0,
        max_hz=60.0,
        peak_height_ratio=0.1,
        peak_min_distance_bins=5,
        top_k=5,
    ),
)
# ==================================================================


def preprocess_like_fbcca(epoch: np.ndarray, input_fs: float, cfg: PreprocessConfig) -> np.ndarray:
    """应用配置化预处理流程。"""
    if epoch.ndim != 2:
        raise ValueError(f"epoch must be 2D (n_channels, n_samples), got shape={epoch.shape}")

    out = epoch.astype(np.float64, copy=True)
    out -= np.mean(out, axis=1, keepdims=True)
    out = detrend(out, axis=1)

    nyquist = 0.5 * float(input_fs)
    hp_cutoff = float(cfg.highpass_cutoff_hz)
    if 0.0 < hp_cutoff < nyquist:
        sos_hp = butter(
            int(cfg.highpass_order),
            hp_cutoff,
            btype="highpass",
            fs=float(input_fs),
            output="sos",
        )
        out = sosfiltfilt(sos_hp, out, axis=1)

    for f0 in cfg.notch_freqs_hz:
        if 0.0 < float(f0) < nyquist:
            b, a = iirnotch(w0=float(f0), Q=float(cfg.notch_q), fs=float(input_fs))
            out = filtfilt(b, a, out, axis=1)

    target_fs = float(cfg.target_fs)
    if abs(target_fs - float(input_fs)) > 1e-6:
        n_samples = max(1, int(round(out.shape[1] * target_fs / float(input_fs))))
        out = resample(out, n_samples, axis=1)

    return out.astype(np.float32, copy=False)


def _parse_channel_excludes(exclude_channels: Sequence[str], total_channels: int) -> Set[int]:
    """将排除通道配置转换为下标集合。"""
    if not exclude_channels:
        return set()

    valid_names = {
        CHANNEL_NAMES[idx].upper(): idx
        for idx in range(min(len(CHANNEL_NAMES), total_channels))
    }

    excluded: Set[int] = set()
    for raw in exclude_channels:
        key = str(raw).strip()
        if not key:
            continue
        if key.isdigit():
            idx = int(key)
            if idx < 0 or idx >= total_channels:
                raise ValueError(f"通道索引越界: {idx}, 当前数据通道数={total_channels}")
            excluded.add(idx)
            continue

        upper_key = key.upper()
        if upper_key not in valid_names:
            raise ValueError(
                f"无效通道名: {key}，可选 {', '.join(CHANNEL_NAMES[:total_channels])} 或 0..{total_channels - 1}"
            )
        excluded.add(valid_names[upper_key])
    return excluded


def _parse_trial_selection(
    n_trials_for_label: int,
    trial_count_per_label: int,
    trial_indices_per_label: Optional[Sequence[int]],
) -> List[int]:
    """返回每个 label 内部的 trial 本地序号（0-based）。"""
    if n_trials_for_label <= 0:
        return []

    if trial_indices_per_label:
        selected = [idx for idx in trial_indices_per_label if 0 <= idx < n_trials_for_label]
        dedup = list(dict.fromkeys(selected))
        return dedup if dedup else [0]

    return list(range(min(max(1, trial_count_per_label), n_trials_for_label)))


def _fft_for_trial(
    trial_data: np.ndarray,
    fs: float,
    preprocess_cfg: PreprocessConfig,
    kept_channel_indices: Sequence[int],
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, float]:
    """对单个 trial 计算频谱。"""
    if preprocess_cfg.enabled:
        trial_data = preprocess_like_fbcca(trial_data, input_fs=fs, cfg=preprocess_cfg)
        actual_fs = float(preprocess_cfg.target_fs)
    else:
        trial_data = trial_data - np.mean(trial_data, axis=1, keepdims=True)
        actual_fs = fs

    trial_data = trial_data[np.array(kept_channel_indices), :]
    n_samples = trial_data.shape[1]
    window = np.hanning(n_samples)

    freqs = fftfreq(n_samples, 1 / actual_fs)
    positive_freq_mask = freqs >= 0
    freqs_positive = freqs[positive_freq_mask]

    fft_results: List[np.ndarray] = []
    for ch in range(trial_data.shape[0]):
        ch_data = trial_data[ch, :] * window
        fft_values = fft(ch_data)
        fft_magnitude = np.abs(fft_values[positive_freq_mask])
        fft_results.append(fft_magnitude)

    avg_spectrum = np.mean(fft_results, axis=0)
    return freqs_positive, fft_results, avg_spectrum, actual_fs


def _get_trial_2d(x_data: np.ndarray, trial_idx: int) -> np.ndarray:
    """
    兼容 trial 数据格式，返回 (n_channels, n_samples) 2D 数组。
    支持:
    - x.shape == (n_trials, n_channels, n_samples)
    - x.shape == (n_trials,), dtype=object, 每个元素是 (n_channels, n_samples)
    """
    trial = x_data[trial_idx]
    trial_arr = np.asarray(trial)
    if trial_arr.ndim != 2:
        raise ValueError(
            f"trial 数据必须是二维 (n_channels, n_samples), got shape={trial_arr.shape}"
        )
    return trial_arr


def _infer_total_channels(x_data: np.ndarray) -> int:
    """从首个 trial 推断通道数。"""
    if len(x_data) == 0:
        raise ValueError("数据为空，无法进行 FFT 分析")
    first_trial = _get_trial_2d(x_data, 0)
    return int(first_trial.shape[0])


def _detect_top_peaks(
    freqs_positive: np.ndarray,
    avg_spectrum: np.ndarray,
    peak_cfg: PeakDetectConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """按配置检测并返回 top-k 峰值。"""
    freq_mask = (freqs_positive >= peak_cfg.min_hz) & (freqs_positive <= peak_cfg.max_hz)
    masked = avg_spectrum[freq_mask]
    if masked.size == 0 or np.max(masked) <= 0:
        return np.array([]), np.array([])

    peaks, _ = find_peaks(
        masked,
        height=np.max(masked) * float(peak_cfg.peak_height_ratio),
        distance=int(peak_cfg.peak_min_distance_bins),
    )
    if peaks.size == 0:
        return np.array([]), np.array([])

    peak_freqs = freqs_positive[freq_mask][peaks]
    peak_mags = masked[peaks]
    top_indices = np.argsort(peak_mags)[::-1][: int(peak_cfg.top_k)]
    return peak_freqs[top_indices], peak_mags[top_indices]


def _average_spectra_on_common_grid(
    spectra_items: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将不同长度/分辨率的频谱插值到公共频率网格后再平均。
    返回: (common_freqs, mean_spectrum)
    """
    if not spectra_items:
        return np.array([]), np.array([])

    base_freqs = spectra_items[0][0]
    max_common_freq = min(freqs[-1] for freqs, _ in spectra_items)
    common_freqs = base_freqs[base_freqs <= max_common_freq]
    if common_freqs.size == 0:
        return np.array([]), np.array([])

    aligned_spectra = []
    for freqs, spectrum in spectra_items:
        aligned = np.interp(common_freqs, freqs, spectrum)
        aligned_spectra.append(aligned)

    return common_freqs, np.mean(np.vstack(aligned_spectra), axis=0)


def analyze_fft(cfg: FFTAnalysisConfig):
    """执行 FFT 分析。"""
    data = np.load(cfg.npy_path, allow_pickle=True).item()
    x = data["x"]
    y = data["y"]
    if not isinstance(x, np.ndarray):
        x = np.asarray(x, dtype=object)
    if not isinstance(y, np.ndarray):
        y = np.asarray(y)

    output_dir = cfg.output_dir_preprocessed if cfg.preprocess.enabled else cfg.output_dir_raw
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_channels = _infer_total_channels(x)
    excluded_channel_indices = _parse_channel_excludes(cfg.exclude_channels, total_channels)
    kept_channel_indices = [i for i in range(total_channels) if i not in excluded_channel_indices]
    if not kept_channel_indices:
        raise ValueError("排除坏道后没有剩余通道可分析")
    kept_channel_names = [CHANNEL_NAMES[i] if i < len(CHANNEL_NAMES) else f"Ch{i}" for i in kept_channel_indices]

    all_peaks: Dict[int, List[Tuple[int, int, List[Tuple[float, float]]]]] = {}

    fig_summary, axes_summary = plt.subplots(2, 4, figsize=(20, 10))
    axes_summary = axes_summary.flatten()

    print("=" * 70)
    print("FFT 频谱分析结果")
    if cfg.preprocess.enabled:
        print("【使用配置化预处理流程】")
        print(
            "预处理参数: "
            f"highpass={cfg.preprocess.highpass_cutoff_hz}Hz(order={cfg.preprocess.highpass_order}), "
            f"notch={cfg.preprocess.notch_freqs_hz}(Q={cfg.preprocess.notch_q}), "
            f"resample={cfg.preprocess.target_fs}Hz"
        )
    else:
        print("【原始数据 (仅去除 DC)】")

    if cfg.trial_indices_per_label:
        print(f"【trial 指定模式】每个 label 分析 trial 序号: {cfg.trial_indices_per_label}")
    else:
        print(f"【trial 数量模式】每个 label 分析前 {cfg.trial_count_per_label} 个 trial")

    if excluded_channel_indices:
        print(f"【排除坏道】{[CHANNEL_NAMES[i] for i in sorted(excluded_channel_indices) if i < len(CHANNEL_NAMES)]}")
    print(f"【参与分析通道】{kept_channel_names}")
    print("=" * 70)
    print(f"采样率: {cfg.fs} Hz")
    print(f"峰值频段: {cfg.peak.min_hz}-{cfg.peak.max_hz} Hz")
    print()

    for label in range(1, 9):
        indices = np.where(y == label)[0]
        if len(indices) == 0:
            print(f"Warning: No trials found for Label {label}")
            continue

        trial_local_indices = _parse_trial_selection(
            n_trials_for_label=len(indices),
            trial_count_per_label=cfg.trial_count_per_label,
            trial_indices_per_label=cfg.trial_indices_per_label,
        )

        all_peaks[label] = []
        summary_spectra: List[Tuple[np.ndarray, np.ndarray]] = []
        target_freq = TARGET_FREQS[label - 1]

        for trial_local_idx in trial_local_indices:
            trial_idx = int(indices[trial_local_idx])
            trial_data = _get_trial_2d(x, trial_idx)
            n_samples = trial_data.shape[1]

            freqs_positive, fft_results, avg_spectrum, actual_fs = _fft_for_trial(
                trial_data=trial_data,
                fs=cfg.fs,
                preprocess_cfg=cfg.preprocess,
                kept_channel_indices=kept_channel_indices,
            )

            top_freqs, top_mags = _detect_top_peaks(freqs_positive, avg_spectrum, cfg.peak)

            all_peaks[label].append(
                (trial_local_idx, trial_idx, list(zip(top_freqs.tolist(), top_mags.tolist())))
            )

            print(
                f"\nLabel {label} (Target: {target_freq} Hz) - "
                f"Trial local/global: {trial_local_idx}/{trial_idx}"
            )
            print(f"  Samples: {n_samples}, Duration: {n_samples / actual_fs:.2f}s")
            print(f"  Top {cfg.peak.top_k} Frequency Peaks:")
            for i, (freq, mag) in enumerate(zip(top_freqs, top_mags), 1):
                marker = " ***" if abs(freq - target_freq) < 1.0 else ""
                print(f"    {i}. {freq:.2f} Hz (magnitude: {mag:.2f}){marker}")

            title_suffix = " (Preprocessed)" if cfg.preprocess.enabled else ""
            n_kept_channels = len(kept_channel_indices)
            fig, axes = plt.subplots(n_kept_channels + 1, 1, figsize=(14, 18), sharex=True)
            fig.suptitle(
                f"Label {label} FFT Spectrum Analysis{title_suffix}\n"
                f"Target Frequency: {target_freq} Hz | Trial local/global: {trial_local_idx}/{trial_idx}",
                fontsize=14,
                fontweight="bold",
            )

            for ch in range(n_kept_channels):
                ax = axes[ch]
                ax.plot(freqs_positive, fft_results[ch], color="steelblue", linewidth=0.8)
                ax.axvline(
                    x=target_freq,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Target: {target_freq}Hz",
                )
                ax.set_ylabel(f"{kept_channel_names[ch]}\\nMag", fontsize=9)
                ax.set_xlim(0, cfg.peak.max_hz)
                ax.grid(True, alpha=0.3)
                ax.legend(loc="upper right", fontsize=7)

            ax_avg = axes[-1]
            ax_avg.plot(
                freqs_positive,
                avg_spectrum,
                color="darkgreen",
                linewidth=1.5,
                label="Average (kept channels)",
            )
            ax_avg.axvline(
                x=target_freq,
                color="red",
                linestyle="--",
                alpha=0.7,
                linewidth=2,
                label=f"Target: {target_freq}Hz",
            )
            for freq, mag in zip(top_freqs[:3], top_mags[:3]):
                ax_avg.plot(freq, mag, "ro", markersize=8)
                ax_avg.annotate(
                    f"{freq:.1f}Hz",
                    xy=(freq, mag),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    color="red",
                )
            ax_avg.set_ylabel("Avg Magnitude", fontsize=10)
            ax_avg.set_xlabel("Frequency (Hz)", fontsize=10)
            ax_avg.set_xlim(0, cfg.peak.max_hz)
            ax_avg.grid(True, alpha=0.3)
            ax_avg.legend(loc="upper right", fontsize=9)

            if len(trial_local_indices) >= 2:
                save_path = (
                    output_path
                    / f"label_{label}_trial_{trial_local_idx:02d}_idx_{trial_idx}_fft_spectrum.png"
                )
            else:
                save_path = output_path / f"label_{label}_fft_spectrum.png"

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {save_path}")
            plt.close(fig)

            summary_spectra.append((freqs_positive, avg_spectrum))

        if summary_spectra:
            summary_freqs, mean_summary_spectrum = _average_spectra_on_common_grid(
                summary_spectra
            )
            if summary_freqs.size == 0:
                continue
            top_freqs, top_mags = _detect_top_peaks(
                summary_freqs, mean_summary_spectrum, cfg.peak
            )

            ax_sum = axes_summary[label - 1]
            ax_sum.plot(summary_freqs, mean_summary_spectrum, color="steelblue", linewidth=1.2)
            ax_sum.axvline(
                x=target_freq,
                color="red",
                linestyle="--",
                alpha=0.7,
                linewidth=2,
                label=f"Target: {target_freq}Hz",
            )
            for freq, mag in zip(top_freqs[:3], top_mags[:3]):
                ax_sum.plot(freq, mag, "ro", markersize=6)
            ax_sum.set_title(f"Label {label} ({target_freq}Hz)", fontsize=12, fontweight="bold")
            ax_sum.set_xlim(0, cfg.peak.max_hz)
            ax_sum.set_xlabel("Frequency (Hz)", fontsize=9)
            ax_sum.set_ylabel("Magnitude", fontsize=9)
            ax_sum.grid(True, alpha=0.3)
            ax_sum.legend(loc="upper right", fontsize=8)

    fig_summary.suptitle("FFT Spectrum Summary - All 8 Labels", fontsize=16, fontweight="bold")
    plt.tight_layout()
    summary_path = output_path / "all_labels_fft_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved summary: {summary_path}")
    plt.close(fig_summary)

    print("\n" + "=" * 70)
    print("频谱峰值汇总")
    print("=" * 70)
    for label in range(1, 9):
        target_freq = TARGET_FREQS[label - 1]
        label_trials = all_peaks.get(label, [])
        print(f"\nLabel {label} (Target: {target_freq} Hz):")
        if not label_trials:
            print("  No significant peaks found")
            continue
        for trial_local_idx, trial_idx, peaks in label_trials:
            print(f"  Trial local/global {trial_local_idx}/{trial_idx}:")
            if peaks:
                for i, (freq, _) in enumerate(peaks, 1):
                    marker = " *** TARGET ***" if abs(freq - target_freq) < 1.0 else ""
                    print(f"    Peak {i}: {freq:.2f} Hz{marker}")
            else:
                print("    No significant peaks found")

    print(f"\n{'=' * 70}")
    print(f"所有图表已保存到: {output_path.absolute()}")
    print("=" * 70)


def _apply_cli_overrides(cfg: FFTAnalysisConfig):
    """允许临时命令行覆盖统一配置区。"""
    import argparse

    parser = argparse.ArgumentParser(description="FFT analysis with centralized config")
    parser.add_argument("--preprocess", "-p", action="store_true", help="启用预处理")
    parser.add_argument("--trials", type=int, help="每个 label 分析前 N 个 trial")
    parser.add_argument("--trial-indices", type=str, help="逗号分隔 trial 序号, 如 0,2,4")
    parser.add_argument("--exclude-channels", type=str, help="逗号分隔坏道, 如 Oz,P3 或 2,6")
    parser.add_argument("--highpass", type=float, help="高通截止频率 Hz")
    parser.add_argument("--highpass-order", type=int, help="高通滤波器阶数")
    parser.add_argument("--notch-freqs", type=str, help="逗号分隔陷波频率, 如 50,100")
    parser.add_argument("--notch-q", type=float, help="陷波 Q 值")
    parser.add_argument("--target-fs", type=float, help="重采样目标频率")
    parser.add_argument("--peak-min-hz", type=float, help="峰值检测最小频率")
    parser.add_argument("--peak-max-hz", type=float, help="峰值检测最大频率")
    parser.add_argument("--peak-ratio", type=float, help="峰值阈值比例")
    parser.add_argument("--peak-distance", type=int, help="峰值最小 bin 间隔")
    parser.add_argument("--peak-topk", type=int, help="输出 top-k 峰值")
    args = parser.parse_args()

    if args.preprocess:
        cfg.preprocess.enabled = True
    if args.trials is not None:
        cfg.trial_count_per_label = max(1, int(args.trials))
    if args.trial_indices:
        cfg.trial_indices_per_label = [
            int(s.strip()) for s in args.trial_indices.split(",") if s.strip()
        ]
    if args.exclude_channels:
        cfg.exclude_channels = [s.strip() for s in args.exclude_channels.split(",") if s.strip()]

    if args.highpass is not None:
        cfg.preprocess.highpass_cutoff_hz = float(args.highpass)
    if args.highpass_order is not None:
        cfg.preprocess.highpass_order = int(args.highpass_order)
    if args.notch_freqs:
        cfg.preprocess.notch_freqs_hz = [
            float(s.strip()) for s in args.notch_freqs.split(",") if s.strip()
        ]
    if args.notch_q is not None:
        cfg.preprocess.notch_q = float(args.notch_q)
    if args.target_fs is not None:
        cfg.preprocess.target_fs = float(args.target_fs)

    if args.peak_min_hz is not None:
        cfg.peak.min_hz = float(args.peak_min_hz)
    if args.peak_max_hz is not None:
        cfg.peak.max_hz = float(args.peak_max_hz)
    if args.peak_ratio is not None:
        cfg.peak.peak_height_ratio = float(args.peak_ratio)
    if args.peak_distance is not None:
        cfg.peak.peak_min_distance_bins = int(args.peak_distance)
    if args.peak_topk is not None:
        cfg.peak.top_k = int(args.peak_topk)


if __name__ == "__main__":
    _apply_cli_overrides(CONFIG)
    analyze_fft(CONFIG)
