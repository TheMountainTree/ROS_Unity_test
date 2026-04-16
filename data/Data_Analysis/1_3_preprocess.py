#!/usr/bin/env python3
"""预处理后分析通道质量：
预处理步骤: 去均值 → 去趋势 → 高通(2Hz) → 50Hz/100Hz 陷波
然后重跑 1_3 的全部分析。
"""

import numpy as np
from scipy.signal import welch, butter, sosfiltfilt, iirnotch, filtfilt, detrend

FREQS = [8.684, 9.706, 11.0, 11.786, 12.692, 13.75, 15.0, 18.333]
LABEL2FREQ = {i + 1: f for i, f in enumerate(FREQS)}
CHANNELS = ["O1", "O2", "Oz", "PO3", "PO4", "Pz", "P3", "P4"]
OCCIPITAL = {"O1", "O2", "Oz"}
FS = 1000

# 预处理参数
HIGHPASS_CUTOFF = 2.0
NOTCH_FREQS = [50.0, 100.0]
NOTCH_Q = 35.0


def preprocess(trial):
    """对单个 trial (n_channels, n_samples) 做预处理。"""
    out = trial.astype(np.float64, copy=True)
    out -= np.mean(out, axis=1, keepdims=True)
    out = detrend(out, axis=1)
    sos_hp = butter(4, HIGHPASS_CUTOFF, btype="highpass", fs=FS, output="sos")
    out = sosfiltfilt(sos_hp, out, axis=1)
    for f0 in NOTCH_FREQS:
        if f0 < FS / 2:
            b, a = iirnotch(w0=f0, Q=NOTCH_Q, fs=FS)
            out = filtfilt(b, a, out, axis=1)
    return out.astype(np.float32)


def main():
    data_path = "data/central_controller_ssvep_node4_test/ssvep4_pretrain_dataset_20260414_161545.npy"
    d = np.load(data_path, allow_pickle=True).item()
    x_raw = d["x"]
    y = d["y"]
    labels = sorted(np.unique(y).tolist())
    n_channels = len(CHANNELS)

    # 预处理所有 trial
    x = [preprocess(trial) for trial in x_raw]

    print("=" * 80)
    print("SSVEP Pretrain Dataset — 预处理后通道质量分析")
    print("=" * 80)
    print(f"预处理: 去均值 → 去趋势(线性) → 高通{HIGHPASS_CUTOFF}Hz → 陷波{NOTCH_FREQS}Hz")
    print(f"通道顺序: {CHANNELS}")
    print(f"枕区通道: {[ch for ch in CHANNELS if ch in OCCIPITAL]}")
    print(f"非枕区通道: {[ch for ch in CHANNELS if ch not in OCCIPITAL]}")
    print()

    # ========== 1. 枕区 vs 非枕区：目标频率 SNR ==========
    print("-" * 80)
    print("1. 枕区 vs 非枕区 — 目标频率 SNR 对比")
    print("-" * 80)

    def compute_snr_ch(trial, freq_hz, ch_idx, nperseg=1024):
        freqs, psd = welch(trial[ch_idx], fs=FS, nperseg=nperseg)
        idx_target = np.argmin(np.abs(freqs - freq_hz))
        mask_neighbor = (np.abs(freqs - freq_hz) <= 2.0) & (np.abs(freqs - freq_hz) > 0.5)
        p_target = psd[idx_target]
        if not np.any(mask_neighbor):
            p_neighbor = np.median(psd) + 1e-12
        else:
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
        for trial in trials:
            for ch_i, ch_name in enumerate(CHANNELS):
                snr = compute_snr_ch(trial, freq_hz, ch_i)
                ch_snr_by_label[ch_name].append(snr)
                if ch_name in OCCIPITAL:
                    occ_vals.append(snr)
                else:
                    non_occ_vals.append(snr)
        occ_snr_all.extend(occ_vals)
        non_occ_snr_all.extend(non_occ_vals)

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

    ch_means = {ch: np.mean(ch_snr_by_label[ch]) for ch in CHANNELS}
    print(f"\n  各通道 SNR 汇总 (所有 label 平均):")
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

    # 识别零方差/极端方差坏导，排除后再算相关性
    dead_channels = set()
    for ch in CHANNELS:
        if var_means[ch] < 1e-6 or var_means[ch] > 1e8:
            dead_channels.add(ch)
    good_indices = [i for i, ch in enumerate(CHANNELS) if ch not in dead_channels]
    good_names = [CHANNELS[i] for i in good_indices]

    print(f"\n  通道间 Pearson 相关矩阵 (排除坏导 {sorted(dead_channels)} 后):")
    n_good = len(good_indices)
    corr_accum = np.zeros((n_good, n_good))
    n_valid = 0
    for trial in all_trials:
        sub = trial[good_indices]
        C = np.corrcoef(sub)
        if np.any(np.isnan(C)):
            continue
        corr_accum += C
        n_valid += 1
    corr_avg = np.zeros((len(CHANNELS), len(CHANNELS)))
    corr_avg[:] = np.nan
    if n_valid > 0:
        sub_avg = corr_accum / n_valid
        for ii, gi in enumerate(good_indices):
            for jj, gj in enumerate(good_indices):
                corr_avg[gi, gj] = sub_avg[ii, jj]

    header = "      " + " ".join(f"{ch:>5}" for ch in CHANNELS)
    print(header)
    for i, ch in enumerate(CHANNELS):
        row = " ".join(f"{corr_avg[i, j]:5.2f}" if not np.isnan(corr_avg[i, j]) else "  nan" for j in range(len(CHANNELS)))
        print(f"  {ch:>4} {row}")

    for ii, ch in enumerate(good_names):
        gi = good_indices[ii]
        off_diag = np.delete(corr_avg[gi, :], gi)
        valid_off = off_diag[~np.isnan(off_diag)]
        if len(valid_off) == 0:
            continue
        mean_corr = np.mean(valid_off)
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
        near_max = np.sum(all_vals > vmax - 0.01 * vrange) / len(all_vals)
        near_min = np.sum(all_vals < vmin + 0.01 * vrange) / len(all_vals)
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
    print("5. 工频污染检测 — 50Hz / 100Hz 功率（陷波后残余）")
    print("-" * 80)

    for ch_i, ch_name in enumerate(CHANNELS):
        snr_50 = []
        snr_100 = []
        for trial in all_trials:
            freqs, psd = welch(trial[ch_i], fs=FS, nperseg=min(len(trial[ch_i]), 2048))
            idx_50 = np.argmin(np.abs(freqs - 50.0))
            idx_100 = np.argmin(np.abs(freqs - 100.0))
            mask_50 = (np.abs(freqs - 50.0) <= 3.0) & (np.abs(freqs - 50.0) > 1.0)
            mask_100 = (np.abs(freqs - 100.0) <= 3.0) & (np.abs(freqs - 100.0) > 1.0)
            ref_50 = np.median(psd[mask_50]) + 1e-12
            ref_100 = np.median(psd[mask_100]) + 1e-12
            snr_50.append(psd[idx_50] / ref_50)
            snr_100.append(psd[idx_100] / ref_100)

        mean_snr50 = np.mean(snr_50)
        mean_snr100 = np.mean(snr_100)
        flag = ""
        if mean_snr50 > 3.0:
            flag += " ⚠ 50Hz残余"
        if mean_snr100 > 3.0:
            flag += " ⚠ 100Hz残余"
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

        # SNR
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
        valid_off = off_diag[~np.isnan(off_diag)]
        if len(valid_off) > 0:
            mean_corr = np.mean(valid_off)
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

        # 工频残余
        snr_50_vals = []
        for trial in all_trials:
            freqs_t, psd_t = welch(trial[ch_i], fs=FS, nperseg=min(len(trial[ch_i]), 2048))
            idx_50 = np.argmin(np.abs(freqs_t - 50.0))
            mask_50 = (np.abs(freqs_t - 50.0) <= 3.0) & (np.abs(freqs_t - 50.0) > 1.0)
            snr_50_vals.append(psd_t[idx_50] / (np.median(psd_t[mask_50]) + 1e-12))
        mean_snr50 = np.mean(snr_50_vals)
        if mean_snr50 > 3.0:
            score -= 2.0
            reasons.append(f"50Hz残余(SNR={mean_snr50:.2f})")

        score = max(0, score)
        scores[ch_name] = score
        reason_str = ", ".join(reasons) if reasons else "无明显问题"
        region = "枕区" if ch_name in OCCIPITAL else "非枕区"
        print(f"    {ch_name:>4} [{region}]: {score:.1f}/10  ({reason_str})")


if __name__ == "__main__":
    main()
