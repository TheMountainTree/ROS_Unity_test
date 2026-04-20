#!/usr/bin/env python3
"""标准 FBCCA 分析（手动坏道剔除版）：
使用所有通道（排除坏道），对 n_harmonics 网格跑 LOO，输出各 label 准确率。

预处理: 手动坏道剔除 → 去均值 → 去趋势 → 带通6-48Hz → 陷波50/100Hz → 降采样256Hz
"""

import sys
import os
import time
import numpy as np
from scipy.signal import resample, butter, sosfiltfilt, iirnotch, filtfilt, detrend

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ssvep_pipeline_fbcca import SSVEPPretrainerFBCCA, SSVEPDecoderFBCCA

# ─── 全局参数 ───
FREQS = [8.684, 9.706, 11.0, 11.786, 12.692, 13.75, 15.0, 18.333]
LABEL2FREQ = {i + 1: f for i, f in enumerate(FREQS)}
ALL_CHANNELS = ["O1", "O2", "Oz", "PO3", "PO4", "Pz", "P3", "P4"]
BAD_CHANNELS = ["O2", "P4", "PO4"]  # 手动坏道剔除，根据1_3分析结果
FS_ORIG = 1000
FS_TARGET = 256

# 预处理参数
BANDPASS_LOW = 6.0
BANDPASS_HIGH = 100.0 #48.0
BANDPASS_ORDER = 4
NOTCH_FREQS = [50.0, 100.0]
NOTCH_Q = 35.0

# FBCCA 参数
FBCCA_SRATE = 256
FBCCA_WP = [(6.0, 50.0), (14.0, 50.0), (22.0, 50.0)]
FBCCA_WS = [(4.0, 52.0), (12.0, 52.0), (20.0, 52.0)]
FBCCA_FILTER_ORDER = 4
FBCCA_RP = 0.5
FBCCA_N_COMPONENTS = 1
FBCCA_N_HARMONICS_GRID = [1, 2, 3, 4, 5]


def detect_bad_channels(x_raw, y):
    """坏道检测：零/极低方差、与其他通道相关性极低。

    在原始数据上检测，避免坏道影响后续去均值和 detrend。

    返回坏道通道名列表。
    """
    n_trials = len(x_raw)
    ch_var = {ch: [] for ch in ALL_CHANNELS}
    for trial in x_raw:
        for ch_i, ch_name in enumerate(ALL_CHANNELS):
            ch_var[ch_name].append(np.var(trial[ch_i].astype(np.float64)))

    var_means = {ch: np.mean(ch_var[ch]) for ch in ALL_CHANNELS}
    var_global = np.mean(list(var_means.values()))
    var_std = np.std(list(var_means.values()))

    bad = []
    # 方差异常：零方差或 z-score > 2
    for ch in ALL_CHANNELS:
        if var_means[ch] < 1e-6:
            bad.append(ch)
            continue
        z = (var_means[ch] - var_global) / (var_std + 1e-12)
        if abs(z) > 2.0:
            bad.append(ch)

    # 相关性异常：排除已标记坏道后，检查剩余通道
    good_indices = [i for i, ch in enumerate(ALL_CHANNELS) if ch not in bad]
    good_names = [ALL_CHANNELS[i] for i in good_indices]
    n_good = len(good_indices)
    if n_good < 2:
        return bad

    corr_accum = np.zeros((n_good, n_good))
    n_valid = 0
    for trial in x_raw:
        sub = trial[good_indices].astype(np.float64)
        C = np.corrcoef(sub)
        if np.any(np.isnan(C)):
            continue
        corr_accum += C
        n_valid += 1

    if n_valid == 0:
        return bad

    corr_avg = corr_accum / n_valid
    for ii, ch in enumerate(good_names):
        off_diag = np.delete(corr_avg[ii, :], ii)
        mean_corr = np.mean(off_diag)
        if mean_corr < 0.3:
            bad.append(ch)

    return sorted(set(bad))


def preprocess(trial, good_indices):
    """对单个 trial 做预处理，只保留 good_indices 对应的通道。

    流程: 坏道已在检测后剔除 → 去均值 → 去趋势 → 带通 → 陷波 → 降采样
    """
    out = trial[good_indices].astype(np.float64, copy=True)
    out -= np.mean(out, axis=1, keepdims=True)
    out = detrend(out, axis=1)

    nyq = FS_ORIG * 0.5
    high = min(BANDPASS_HIGH, nyq - 0.5)
    if high > BANDPASS_LOW:
        sos_bp = butter(
            BANDPASS_ORDER, [BANDPASS_LOW, high], btype="bandpass", fs=FS_ORIG, output="sos"
        )
        out = sosfiltfilt(sos_bp, out, axis=1)

    for f0 in NOTCH_FREQS:
        if f0 < FS_ORIG / 2:
            b, a = iirnotch(w0=f0, Q=NOTCH_Q, fs=FS_ORIG)
            out = filtfilt(b, a, out, axis=1)

    return out


def resample_epoch(epoch, target_fs=FS_TARGET):
    n_target = int(epoch.shape[1] * target_fs / FS_ORIG)
    return resample(epoch, n_target, axis=1)


def prepare_data():
    """加载数据，手动坏道剔除，预处理，返回 X, y, good_channels。"""
    data_path = "data/central_controller_ssvep_node4_test/ssvep4_pretrain_dataset_20260416_164519.npy"
    d = np.load(data_path, allow_pickle=True).item()
    x_raw = d["x"]
    y = d["y"]

    # 手动坏道剔除
    bad = sorted(set(BAD_CHANNELS) & set(ALL_CHANNELS))
    good_channels = [ch for ch in ALL_CHANNELS if ch not in bad]
    good_indices = [ALL_CHANNELS.index(ch) for ch in good_channels]

    print(f"手动坏道: {bad if bad else '无'}")
    print(f"使用通道: {good_channels}")

    # 预处理
    epochs = []
    for trial in x_raw:
        pp = preprocess(trial, good_indices)
        rs = resample_epoch(pp)
        epochs.append(rs)

    min_len = min(e.shape[1] for e in epochs)
    epochs = [e[:, :min_len] for e in epochs]

    X = np.stack(epochs, axis=0).astype(np.float64)
    y = np.array(y, dtype=np.int32)
    return X, y, good_channels


def loo_evaluate(X, y, n_harmonics=5):
    """标准 LOO：逐 trial 留出，输出总体 + 各 label 准确率。"""
    n = len(y)
    correct = 0
    per_label = {}

    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        train_labels = np.sort(np.unique(y_train))
        X_test = X[i:i + 1]
        y_test = y[i]
        try:
            pt = SSVEPPretrainerFBCCA(
                srate=FBCCA_SRATE, wp=FBCCA_WP, ws=FBCCA_WS,
                filter_order=FBCCA_FILTER_ORDER, rp=FBCCA_RP,
                n_components=FBCCA_N_COMPONENTS, n_harmonics=n_harmonics,
                freqs=FREQS,
            )
            pt.fit(X_train, y_train)
            dec = SSVEPDecoderFBCCA(pt)
            pred_idx = int(dec.decode(X_test)[0])
            # MetaBCI FBSCCA returns 0-based class index (argmax over references).
            # Map index back to original training labels before accuracy counting.
            if 0 <= pred_idx < len(train_labels):
                pred = int(train_labels[pred_idx])
            else:
                pred = -1
        except Exception as e:
            print(f"    LOO trial {i}: error {e}")
            pred = -1
        if pred == y_test:
            correct += 1
        label = int(y_test)
        if label not in per_label:
            per_label[label] = {"correct": 0, "total": 0}
        per_label[label]["total"] += 1
        if pred == y_test:
            per_label[label]["correct"] += 1

    acc = correct / n
    return acc, per_label


def main():
    print("=" * 80)
    print("标准 FBCCA (FBSCCA) LOO 分析")
    print("=" * 80)
    print(f"预处理: 手动坏道剔除{BAD_CHANNELS} → 去均值 → 去趋势 → 带通{BANDPASS_LOW}-{BANDPASS_HIGH}Hz → "
          f"陷波{NOTCH_FREQS}Hz → 降采样{FS_TARGET}Hz")
    print(f"FBCCA: srate={FBCCA_SRATE}, wp={FBCCA_WP}, "
          f"n_components={FBCCA_N_COMPONENTS}, n_harmonics_grid={FBCCA_N_HARMONICS_GRID}")
    print(f"目标频率: {FREQS}")
    print()

    # ─── 数据准备 ───
    X, y, good_channels = prepare_data()
    print(f"  X.shape={X.shape}, y unique={np.unique(y).tolist()}")
    print()

    # ─── LOO 评估 (n_harmonics 网格) ───
    all_results = {}
    for nh in FBCCA_N_HARMONICS_GRID:
        print(f"--- n_harmonics={nh} ---")
        t0 = time.time()
        acc, per_label = loo_evaluate(X, y, n_harmonics=nh)
        elapsed = time.time() - t0

        print(f"  总准确率: {acc:.4f}  耗时: {elapsed:.1f}s")
        print(f"  各 Label 准确率:")
        for label in sorted(per_label.keys()):
            p = per_label[label]
            la = p["correct"] / p["total"]
            freq = LABEL2FREQ[label]
            print(f"    Label {label} ({freq:>5.1f}Hz): {p['correct']}/{p['total']} = {la:.4f}")
        print()

        all_results[nh] = {"accuracy": acc, "per_label": per_label}

    # ─── 汇总 ───
    print("=" * 80)
    print("汇总")
    print("=" * 80)

    header = f"{'n_harmonics':>10} {'总准确率':>8}"
    for label in sorted(LABEL2FREQ.keys()):
        header += f" {'L'+str(label)+'('+str(int(LABEL2FREQ[label]))+'Hz)':>10}"
    print(header)
    print("-" * len(header))

    for nh in FBCCA_N_HARMONICS_GRID:
        r = all_results[nh]
        row = f"{nh:>10} {r['accuracy']:>8.4f}"
        for label in sorted(LABEL2FREQ.keys()):
            if label in r["per_label"]:
                p = r["per_label"][label]
                la = p["correct"] / p["total"]
                row += f" {la:>10.4f}"
            else:
                row += f" {'N/A':>10}"
        print(row)

    print()
    print(f"使用通道: {good_channels}")


if __name__ == "__main__":
    main()
