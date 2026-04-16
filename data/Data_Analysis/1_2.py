#!/usr/bin/env python3
"""分析预训练 dataset 的频谱质量与 SNR：
每个 label 的 trial 在目标频率处是否有明显峰值，是否能看到合理谐波。
"""

import numpy as np
from scipy.signal import welch

FREQS = [8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 40.0, 45.0]
LABEL2FREQ = {i + 1: f for i, f in enumerate(FREQS)}
CHANNELS = ["O1", "O2", "Oz", "PO3", "PO4", "Pz", "P3", "P4"]
FS = 1000  # 原始采样率
NYQUIST = FS / 2


def compute_psd(epoch, fs=FS, nperseg=512):
    """计算单次 trial 的 PSD（所有通道平均）。

    Returns:
        freqs: ndarray
        psd: ndarray, shape (n_channels, n_freqs)
    """
    freqs, psd = welch(epoch, fs=fs, nperseg=nperseg, axis=1)
    return freqs, psd


def snr_at_freq(psd, freqs, target_hz, bandwidth=2.0):
    """计算目标频率处的 SNR。

    SNR = PSD(f_target) / median(PSD(f_target ± bandwidth, 排除 f_target 附近 ±1Hz))
    """
    # 目标频率处的功率
    idx_target = np.argmin(np.abs(freqs - target_hz))
    p_target = np.mean(psd[:, idx_target])

    # 邻域功率（排除目标频率 ±1Hz）
    mask_neighbor = (np.abs(freqs - target_hz) <= bandwidth) & (np.abs(freqs - target_hz) > 1.0)
    if not np.any(mask_neighbor):
        return p_target / (np.median(psd) + 1e-12)
    p_neighbor = np.median(psd[:, mask_neighbor])
    return p_target / (p_neighbor + 1e-12)


def find_harmonics(freq_hz, max_freq=NYQUIST, n_max=5):
    """返回目标频率的前 n_max 阶谐波（不超过 max_freq）。"""
    harmonics = []
    for n in range(2, n_max + 1):
        hf = freq_hz * n
        if hf < max_freq:
            harmonics.append((n, hf))
    return harmonics


def main():
    data_path = "data/central_controller_ssvep_node4_test/ssvep4_pretrain_dataset_20260409_162235.npy"
    d = np.load(data_path, allow_pickle=True).item()
    x = d["x"]
    y = d["y"]
    labels = sorted(np.unique(y).tolist())

    print("=" * 80)
    print("SSVEP Pretrain Dataset — 频谱质量 / SNR 分析")
    print("=" * 80)
    print(f"数据集: {data_path}")
    print(f"目标频率: {FREQS}")
    print(f"通道: {CHANNELS}")
    print()

    # ========== 1. 各 label 的 SNR ==========
    print("-" * 80)
    print("1. 各 Label 目标频率 SNR（PSD_peak / PSD_neighbor）")
    print("-" * 80)

    snr_summary = {}
    for label in labels:
        freq_hz = LABEL2FREQ[label]
        trials = [x[i] for i in range(len(y)) if y[i] == label]
        n_trials = len(trials)

        freqs, _ = compute_psd(trials[0])
        snr_trials = []
        for trial in trials:
            _, psd = compute_psd(trial)
            snr_val = snr_at_freq(psd, freqs, freq_hz)
            snr_trials.append(snr_val)

        snr_arr = np.array(snr_trials)
        snr_summary[label] = snr_arr
        print(f"  Label {label} ({freq_hz:>5.1f}Hz): "
              f"SNR mean={np.mean(snr_arr):.2f}, std={np.std(snr_arr):.2f}, "
              f"min={np.min(snr_arr):.2f}, max={np.max(snr_arr):.2f}")

    print()

    # ========== 2. 通道级 SNR 对比 ==========
    print("-" * 80)
    print("2. 各通道在目标频率处的 SNR（所有 trial 平均）")
    print("-" * 80)

    ch_snr = {ch: [] for ch in CHANNELS}
    for label in labels:
        freq_hz = LABEL2FREQ[label]
        trials = [x[i] for i in range(len(y)) if y[i] == label]
        for trial in trials:
            freqs, psd = compute_psd(trial)
            idx_target = np.argmin(np.abs(freqs - freq_hz))
            mask_neighbor = (np.abs(freqs - freq_hz) <= 2.0) & (np.abs(freqs - freq_hz) > 1.0)
            p_neighbor = np.median(psd[:, mask_neighbor], axis=1) + 1e-12
            for ch_i, ch_name in enumerate(CHANNELS):
                ch_snr[ch_name].append(psd[ch_i, idx_target] / p_neighbor[ch_i])

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
                mask_neighbor = (np.abs(freqs - freq_hz) <= 2.0) & (np.abs(freqs - freq_hz) > 1.0)
                p_neighbor = np.median(psd[:, mask_neighbor], axis=1) + 1e-12
                vals.append(psd[ch_i, idx_target] / p_neighbor[ch_i])
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
        # 所有 trial PSD 平均
        all_psd = []
        for trial in trials:
            freqs, psd = compute_psd(trial)
            all_psd.append(psd)
        avg_psd = np.mean(all_psd, axis=0)  # (n_channels, n_freqs)

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

    # ========== 4. 频谱峰值可视化（文本） ==========
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

        # 归一化
        avg_psd_oz_norm = avg_psd_oz / (np.median(avg_psd_oz) + 1e-12)
        # 找 top-5 峰值
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
        # 标记对角线
        diag_str = ""
        for i, label_col in enumerate(labels):
            if label_col == label_row:
                # 用 * 标记自身频率
                pass
        print(row_str + "  ← 目标=" + f"{freq_row:.1f}Hz")


if __name__ == "__main__":
    main()
