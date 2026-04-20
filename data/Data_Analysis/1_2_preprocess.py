#!/usr/bin/env python3
"""预处理后分析频谱质量与 SNR：
预处理步骤: 去均值 → 去趋势 → 高通(2Hz) → 50Hz/100Hz 陷波
然后重跑 1_2 的全部分析。
"""

import numpy as np
from scipy.signal import welch, butter, sosfiltfilt, iirnotch, filtfilt, detrend

FREQS = [8.684, 9.706, 11.0, 11.786, 12.692, 13.75, 15.0, 18.333]
LABEL2FREQ = {i + 1: f for i, f in enumerate(FREQS)}
CHANNELS = ["O1", "O2", "Oz", "PO3", "PO4", "Pz", "P3", "P4"]
FS = 1000
NYQUIST = FS / 2

# 预处理参数
HIGHPASS_CUTOFF = 6.0   # Hz
NOTCH_FREQS = [50.0, 100.0]
NOTCH_Q = 35.0


def preprocess(trial):
    """对单个 trial (n_channels, n_samples) 做预处理。

    Steps:
      1. 去均值 (每通道)
      2. 去趋势 (线性)
      3. 高通 2Hz (4阶 Butterworth)
      4. 50Hz + 100Hz 陷波
    """
    out = trial.astype(np.float64, copy=True)

    # 1. 去均值
    out -= np.mean(out, axis=1, keepdims=True)

    # 2. 去趋势
    out = detrend(out, axis=1)

    # 3. 高通滤波
    sos_hp = butter(4, HIGHPASS_CUTOFF, btype="highpass", fs=FS, output="sos")
    out = sosfiltfilt(sos_hp, out, axis=1)

    # 4. 陷波
    for f0 in NOTCH_FREQS:
        if f0 < NYQUIST:
            b, a = iirnotch(w0=f0, Q=NOTCH_Q, fs=FS)
            out = filtfilt(b, a, out, axis=1)

    return out.astype(np.float32)


def compute_psd(epoch, fs=FS, nperseg=1024):
    freqs, psd = welch(epoch, fs=fs, nperseg=nperseg, axis=1)
    return freqs, psd


def snr_at_freq(psd, freqs, target_hz, bandwidth=2.0, exclude_radius=0.5):
    idx_target = np.argmin(np.abs(freqs - target_hz))
    p_target = np.mean(psd[:, idx_target])
    mask_neighbor = (np.abs(freqs - target_hz) <= bandwidth) & (np.abs(freqs - target_hz) > exclude_radius)
    if not np.any(mask_neighbor):
        return p_target / (np.median(psd) + 1e-12)
    p_neighbor = np.median(psd[:, mask_neighbor])
    return p_target / (p_neighbor + 1e-12)


def find_harmonics(freq_hz, max_freq=NYQUIST, n_max=5):
    harmonics = []
    for n in range(2, n_max + 1):
        hf = freq_hz * n
        if hf < max_freq:
            harmonics.append((n, hf))
    return harmonics


def main():
    data_path = "data/central_controller_ssvep_node4_test/ssvep4_pretrain_dataset_20260416_164519.npy"
    d = np.load(data_path, allow_pickle=True).item()
    x_raw = d["x"]
    y = d["y"]
    labels = sorted(np.unique(y).tolist())

    # 预处理所有 trial
    x = [preprocess(trial) for trial in x_raw]

    print("=" * 80)
    print("SSVEP Pretrain Dataset — 预处理后频谱质量 / SNR 分析")
    print("=" * 80)
    print(f"预处理: 去均值 → 去趋势(线性) → 高通{HIGHPASS_CUTOFF}Hz → 陷波{NOTCH_FREQS}Hz")
    print(f"数据集: {data_path}")
    print(f"目标频率: {FREQS}")
    print(f"通道: {CHANNELS}")
    print()

    # ========== 1. 各 label 的 SNR ==========
    print("-" * 80)
    print("1. 各 Label 目标频率 SNR（PSD_peak / PSD_neighbor）")
    print("-" * 80)

    for label in labels:
        freq_hz = LABEL2FREQ[label]
        trials = [x[i] for i in range(len(y)) if y[i] == label]
        snr_trials = []
        for trial in trials:
            freqs, psd = compute_psd(trial)
            snr_val = snr_at_freq(psd, freqs, freq_hz)
            snr_trials.append(snr_val)
        snr_arr = np.array(snr_trials)
        print(f"  Label {label} ({freq_hz:>5.1f}Hz): "
              f"SNR mean={np.mean(snr_arr):.2f}, std={np.std(snr_arr):.2f}, "
              f"min={np.min(snr_arr):.2f}, max={np.max(snr_arr):.2f}")

    print()

    # ========== 2. 通道级 SNR 对比 ==========
    print("-" * 80)
    print("2. 各通道在目标频率处的 SNR（所有 trial 平均）")
    print("-" * 80)

    header = f"{'Label':>6} {'Freq':>6}" + "".join(f" {ch:>6}" for ch in CHANNELS)
    print(header)
    for label in labels:
        freq_hz = LABEL2FREQ[label]
        trials = [x[i] for i in range(len(y)) if y[i] == label]
        ch_means = []
        for ch_i, ch_name in enumerate(CHANNELS):
            vals = []
            for trial in trials:
                freqs, psd = compute_psd(trial)
                idx_target = np.argmin(np.abs(freqs - freq_hz))
                mask_neighbor = (np.abs(freqs - freq_hz) <= 2.0) & (np.abs(freqs - freq_hz) > 0.5)
                if not np.any(mask_neighbor):
                    p_neighbor_ch = np.median(psd[ch_i]) + 1e-12
                else:
                    p_neighbor_ch = np.median(psd[ch_i, mask_neighbor]) + 1e-12
                vals.append(psd[ch_i, idx_target] / p_neighbor_ch)
            ch_means.append(np.mean(vals))
        row = f"  {label:>4} {freq_hz:>5.1f}" + "".join(f" {v:>6.2f}" for v in ch_means)
        print(row)

    print()

    # ========== 3. 谐波检测 ==========
    print("-" * 80)
    print("3. 谐波检测 — 目标频率的倍频处是否有 SNR 峰值")
    print("-" * 80)

    for label in labels:
        freq_hz = LABEL2FREQ[label]
        trials = [x[i] for i in range(len(y)) if y[i] == label]
        all_psd = []
        for trial in trials:
            freqs, psd = compute_psd(trial)
            all_psd.append(psd)
        avg_psd = np.mean(all_psd, axis=0)

        harmonics = find_harmonics(freq_hz)
        if not harmonics:
            print(f"  Label {label} ({freq_hz:.1f}Hz): 无谐波 < {NYQUIST}Hz")
            continue

        h_str_parts = [f"基频 {freq_hz:.1f}Hz: SNR={snr_at_freq(avg_psd, freqs, freq_hz):.2f}"]
        for n, hf in harmonics:
            h_snr = snr_at_freq(avg_psd, freqs, hf)
            h_str_parts.append(f"  {n}次谐波 {hf:.1f}Hz: SNR={h_snr:.2f}")
        print(f"  Label {label}: " + " |".join(h_str_parts))

    print()

    # ========== 4. 频谱峰值可视化 ==========
    print("-" * 80)
    print("4. 频谱概览 — 各 label 在 Oz 通道的归一化 PSD 顶部频率")
    print("-" * 80)

    oz_idx = CHANNELS.index("Oz")
    for label in labels:
        freq_hz = LABEL2FREQ[label]
        trials = [x[i] for i in range(len(y)) if y[i] == label]
        all_psd = []
        for trial in trials:
            freqs, psd = compute_psd(trial)
            all_psd.append(psd[oz_idx])
        avg_psd_oz = np.mean(all_psd, axis=0)

        avg_psd_oz_norm = avg_psd_oz / (np.median(avg_psd_oz) + 1e-12)
        top5_idx = np.argsort(avg_psd_oz_norm)[-5:][::-1]
        top5_str = ", ".join(f"{freqs[i]:.1f}Hz(x{avg_psd_oz_norm[i]:.1f})" for i in top5_idx)
        print(f"  Label {label} ({freq_hz:.1f}Hz) Oz top-5: {top5_str}")

    print()

    # ========== 5. 类间频谱区分度 ==========
    print("-" * 80)
    print("5. 类间频谱区分度 — 每个 label 在非自身目标频率处的 SNR（越低越好）")
    print("-" * 80)

    print(f"{'':>6}" + "".join(f" L{l:>1}" for l in labels))
    for label_row in labels:
        freq_row = LABEL2FREQ[label_row]
        trials = [x[i] for i in range(len(y)) if y[i] == label_row]
        all_psd = []
        for trial in trials:
            freqs, psd = compute_psd(trial)
            all_psd.append(psd)
        avg_psd = np.mean(all_psd, axis=0)

        row_vals = []
        for label_col in labels:
            freq_col = LABEL2FREQ[label_col]
            s = snr_at_freq(avg_psd, freqs, freq_col)
            row_vals.append(s)
        row_str = f"  L{label_row:>1} " + "".join(f" {v:>4.1f}" for v in row_vals)
        print(row_str + f"  ← 目标={freq_row:.1f}Hz")


if __name__ == "__main__":
    main()
