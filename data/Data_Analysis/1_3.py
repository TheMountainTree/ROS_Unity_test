#!/usr/bin/env python3
"""分析预训练 dataset 的通道质量：
1. 枕区(O1/Oz/O2) vs 非视觉区通道的目标频率 SNR 对比
2. 坏导检测（方差异常、相关性异常）
3. 饱和检测（信号削波）
4. 漂移检测（超低频功率占比）
5. 工频污染检测（50Hz/100Hz 功率）
"""

import numpy as np
from scipy.signal import welch

FREQS = [8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 40.0, 45.0]
LABEL2FREQ = {i + 1: f for i, f in enumerate(FREQS)}
CHANNELS = ["O1", "O2", "Oz", "PO3", "PO4", "Pz", "P3", "P4"]
OCCIPITAL = {"O1", "O2", "Oz"}
FS = 1000


def main():
    data_path = "data/central_controller_ssvep_node4_test/ssvep4_pretrain_dataset_20260409_162235.npy"
    d = np.load(data_path, allow_pickle=True).item()
    x = d["x"]
    y = d["y"]
    labels = sorted(np.unique(y).tolist())
    n_channels = len(CHANNELS)

    print("=" * 80)
    print("SSVEP Pretrain Dataset — 通道质量分析")
    print("=" * 80)
    print(f"通道顺序: {CHANNELS}")
    print(f"枕区通道: {[ch for ch in CHANNELS if ch in OCCIPITAL]}")
    print(f"非枕区通道: {[ch for ch in CHANNELS if ch not in OCCIPITAL]}")
    print()

    # ========== 1. 枕区 vs 非枕区：目标频率 SNR ==========
    print("-" * 80)
    print("1. 枕区 vs 非枕区 — 目标频率 SNR 对比")
    print("-" * 80)

    def compute_snr_ch(trial, freq_hz, ch_idx, nperseg=512):
        freqs, psd = welch(trial[ch_idx], fs=FS, nperseg=nperseg)
        idx_target = np.argmin(np.abs(freqs - freq_hz))
        mask_neighbor = (np.abs(freqs - freq_hz) <= 2.0) & (np.abs(freqs - freq_hz) > 1.0)
        p_target = psd[idx_target]
        p_neighbor = np.median(psd[mask_neighbor]) + 1e-12
        return p_target / p_neighbor

    occ_snr_all = []
    non_occ_snr_all = []
    ch_snr_by_label = {ch: [] for ch in CHANNELS}

    for label in labels:
        freq_hz = LABEL2FREQ[label]
        trials = [x[i] for i in range(len(y)) if y[i] == label]
        occ_vals = []
        non_occ_vals = []
        ch_vals = {ch: [] for ch in CHANNELS}
        for trial in trials:
            for ch_i, ch_name in enumerate(CHANNELS):
                snr = compute_snr_ch(trial, freq_hz, ch_i)
                ch_vals[ch_name].append(snr)
                if ch_name in OCCIPITAL:
                    occ_vals.append(snr)
                else:
                    non_occ_vals.append(snr)
        occ_snr_all.extend(occ_vals)
        non_occ_snr_all.extend(non_occ_vals)
        for ch_name in CHANNELS:
            ch_snr_by_label[ch_name].extend(ch_vals[ch_name])

        occ_mean = np.mean(occ_vals)
        non_occ_mean = np.mean(non_occ_vals)
        ratio = occ_mean / (non_occ_mean + 1e-12)
        print(f"  Label {label} ({freq_hz:>5.1f}Hz): "
              f"枕区 mean SNR={occ_mean:.2f}, 非枕区 mean SNR={non_occ_mean:.2f}, "
              f"枕区/非枕区={ratio:.2f}")

    occ_global = np.mean(occ_snr_all)
    non_occ_global = np.mean(non_occ_snr_all)
    print(f"\n  全局: 枕区 mean SNR={occ_global:.2f}, 非枕区 mean SNR={non_occ_global:.2f}, "
          f"枕区/非枕区={occ_global / (non_occ_global + 1e-12):.2f}")

    # 各通道汇总
    print(f"\n  各通道 SNR 汇总 (所有 label 平均):")
    ch_means = {ch: np.mean(ch_snr_by_label[ch]) for ch in CHANNELS}
    for ch in CHANNELS:
        marker = "★" if ch in OCCIPITAL else " "
        print(f"    {marker} {ch:>4}: mean SNR={ch_means[ch]:.2f}")

    print()

    # ========== 2. 坏导检测 ==========
    print("-" * 80)
    print("2. 坏导检测 — 方差异常 / 与其他通道相关性异常")
    print("-" * 80)

    all_trials = list(x)
    ch_var = {ch: [] for ch in CHANNELS}
    for trial in all_trials:
        for ch_i, ch_name in enumerate(CHANNELS):
            ch_var[ch_name].append(np.var(trial[ch_i]))

    var_means = {ch: np.mean(ch_var[ch]) for ch in CHANNELS}
    var_global = np.mean(list(var_means.values()))
    var_std = np.std(list(var_means.values()))

    print(f"  通道方差 (均值):")
    bad_channels = []
    for ch in CHANNELS:
        z = (var_means[ch] - var_global) / (var_std + 1e-12)
        flag = ""
        if abs(z) > 2.0:
            flag = " ⚠ 方差异常"
            bad_channels.append(ch)
        print(f"    {ch:>4}: var={var_means[ch]:.4e}  z-score={z:+.2f}{flag}")

    # 通道间相关性
    print(f"\n  通道间 Pearson 相关矩阵 (所有 trial 展平平均):")
    n_ch = len(CHANNELS)
    corr_accum = np.zeros((n_ch, n_ch))
    for trial in all_trials:
        # 展平时间轴
        flat = trial  # (n_ch, n_samples)
        C = np.corrcoef(flat)
        corr_accum += C
    corr_avg = corr_accum / len(all_trials)

    header = "      " + " ".join(f"{ch:>5}" for ch in CHANNELS)
    print(header)
    for i, ch in enumerate(CHANNELS):
        row = " ".join(f"{corr_avg[i, j]:5.2f}" for j in range(n_ch))
        print(f"  {ch:>4} {row}")

    # 找与其他通道相关性极低的通道
    for i, ch in enumerate(CHANNELS):
        off_diag = np.delete(corr_avg[i, :], i)
        mean_corr = np.mean(off_diag)
        if mean_corr < 0.3:
            print(f"    ⚠ {ch} 与其他通道平均相关性仅 {mean_corr:.3f}，疑似坏导")
            bad_channels.append(ch)

    if bad_channels:
        bad_channels = sorted(set(bad_channels))
        print(f"\n  疑似坏导通道: {bad_channels}")
    else:
        print(f"\n  未检测到明显坏导")

    print()

    # ========== 3. 饱和检测 ==========
    print("-" * 80)
    print("3. 饱和/削波检测 — 信号值是否触及 A/D 上下限")
    print("-" * 80)

    for ch_i, ch_name in enumerate(CHANNELS):
        all_vals = np.concatenate([trial[ch_i] for trial in all_trials])
        vmin, vmax = np.min(all_vals), np.max(all_vals)
        vrange = vmax - vmin
        # 检查是否有大量样本落在极值 ±1% 范围内
        near_max = np.sum(all_vals > vmax - 0.01 * vrange) / len(all_vals)
        near_min = np.sum(all_vals < vmin + 0.01 * vrange) / len(all_vals)
        # 检查唯一值数量（饱和信号离散值少）
        n_unique = len(np.unique(all_vals))
        flag = ""
        if near_max > 0.05 or near_min > 0.05:
            flag = f" ⚠ 极值附近样本过多 (top={near_max:.3f}, bot={near_min:.3f})"
        print(f"    {ch_name:>4}: min={vmin:.2f}, max={vmax:.2f}, "
              f"range={vrange:.2f}, unique_vals={n_unique}{flag}")

    print()

    # ========== 4. 漂移检测 ==========
    print("-" * 80)
    print("4. 漂移检测 — 超低频 (<2Hz) 功率占总功率比例")
    print("-" * 80)

    for ch_i, ch_name in enumerate(CHANNELS):
        drift_ratios = []
        for trial in all_trials:
            freqs, psd = welch(trial[ch_i], fs=FS, nperseg=min(len(trial[ch_i]), 2048))
            total_power = np.sum(psd)
            drift_power = np.sum(psd[freqs < 2.0])
            drift_ratios.append(drift_power / (total_power + 1e-12))
        mean_drift = np.mean(drift_ratios)
        flag = ""
        if mean_drift > 0.5:
            flag = " ⚠ 严重漂移"
        elif mean_drift > 0.3:
            flag = " ⚠ 中度漂移"
        print(f"    {ch_name:>4}: <2Hz 功率占比={mean_drift:.4f}{flag}")

    print()

    # ========== 5. 工频污染检测 ==========
    print("-" * 80)
    print("5. 工频污染检测 — 50Hz / 100Hz 功率")
    print("-" * 80)

    for ch_i, ch_name in enumerate(CHANNELS):
        power_50 = []
        power_100 = []
        snr_50 = []
        snr_100 = []
        for trial in all_trials:
            freqs, psd = welch(trial[ch_i], fs=FS, nperseg=min(len(trial[ch_i]), 2048))
            idx_50 = np.argmin(np.abs(freqs - 50.0))
            idx_100 = np.argmin(np.abs(freqs - 100.0))
            # 邻域参考
            mask_50 = (np.abs(freqs - 50.0) <= 3.0) & (np.abs(freqs - 50.0) > 1.0)
            mask_100 = (np.abs(freqs - 100.0) <= 3.0) & (np.abs(freqs - 100.0) > 1.0)
            ref_50 = np.median(psd[mask_50]) + 1e-12
            ref_100 = np.median(psd[mask_100]) + 1e-12
            power_50.append(psd[idx_50])
            power_100.append(psd[idx_100])
            snr_50.append(psd[idx_50] / ref_50)
            snr_100.append(psd[idx_100] / ref_100)

        mean_snr50 = np.mean(snr_50)
        mean_snr100 = np.mean(snr_100)
        flag = ""
        if mean_snr50 > 3.0:
            flag += " ⚠ 50Hz污染"
        if mean_snr100 > 3.0:
            flag += " ⚠ 100Hz污染"
        print(f"    {ch_name:>4}: 50Hz SNR={mean_snr50:.2f}, 100Hz SNR={mean_snr100:.2f}{flag}")

    print()

    # ========== 6. 综合通道质量评分 ==========
    print("-" * 80)
    print("6. 综合通道质量评分 (0-10, 越高越好)")
    print("-" * 80)

    scores = {}
    for ch_i, ch_name in enumerate(CHANNELS):
        score = 10.0
        reasons = []

        # SNR 评分
        snr = ch_means[ch_name]
        if snr < 1.5:
            score -= 3.0
            reasons.append(f"SNR低({snr:.2f})")
        elif snr < 2.0:
            score -= 1.5
            reasons.append(f"SNR偏低({snr:.2f})")

        # 方差异常
        z = (var_means[ch_name] - var_global) / (var_std + 1e-12)
        if abs(z) > 2.0:
            score -= 2.0
            reasons.append(f"方差异常(z={z:.2f})")

        # 相关性
        off_diag = np.delete(corr_avg[ch_i, :], ch_i)
        mean_corr = np.mean(off_diag)
        if mean_corr < 0.3:
            score -= 2.0
            reasons.append(f"低相关({mean_corr:.3f})")
        elif mean_corr < 0.5:
            score -= 1.0
            reasons.append(f"相关偏低({mean_corr:.3f})")

        # 漂移
        drift_vals = []
        for trial in all_trials:
            freqs_t, psd_t = welch(trial[ch_i], fs=FS, nperseg=min(len(trial[ch_i]), 2048))
            drift_vals.append(np.sum(psd_t[freqs_t < 2.0]) / (np.sum(psd_t) + 1e-12))
        mean_drift = np.mean(drift_vals)
        if mean_drift > 0.5:
            score -= 2.0
            reasons.append(f"严重漂移({mean_drift:.3f})")
        elif mean_drift > 0.3:
            score -= 1.0
            reasons.append(f"中度漂移({mean_drift:.3f})")

        # 工频
        snr_50_vals = []
        for trial in all_trials:
            freqs_t, psd_t = welch(trial[ch_i], fs=FS, nperseg=min(len(trial[ch_i]), 2048))
            idx_50 = np.argmin(np.abs(freqs_t - 50.0))
            mask_50 = (np.abs(freqs_t - 50.0) <= 3.0) & (np.abs(freqs_t - 50.0) > 1.0)
            snr_50_vals.append(psd_t[idx_50] / (np.median(psd_t[mask_50]) + 1e-12))
        mean_snr50 = np.mean(snr_50_vals)
        if mean_snr50 > 3.0:
            score -= 2.0
            reasons.append(f"50Hz污染(SNR={mean_snr50:.2f})")

        score = max(0, score)
        scores[ch_name] = score
        reason_str = ", ".join(reasons) if reasons else "无明显问题"
        region = "枕区" if ch_name in OCCIPITAL else "非枕区"
        print(f"    {ch_name:>4} [{region}]: {score:.1f}/10  ({reason_str})")


if __name__ == "__main__":
    main()
