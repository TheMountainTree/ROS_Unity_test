#!/usr/bin/env python3
"""标准 FBCCA LOO 分析（固定裁剪版）：
通道: O1, O2, Oz, PO3
固定裁掉前0.5s后，执行标准 LOO 并输出各 label 准确率。

预处理: 坏道检测 → 去均值 → 去趋势 → 带通6-48Hz → 陷波50/100Hz → 固定裁掉前0.5s → 降采样256Hz
"""

import sys
import os
import time
import numpy as np
from scipy.signal import resample, butter, sosfiltfilt, iirnotch, filtfilt, detrend

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ssvep_pipeline_fbcca import SSVEPPretrainerFBCCA, SSVEPDecoderFBCCA

# ─── 全局参数 ───
FREQS = [8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 40.0, 45.0]
LABEL2FREQ = {i + 1: f for i, f in enumerate(FREQS)}
ALL_CHANNELS = ["O1", "O2", "Oz", "PO3", "PO4", "Pz", "P3", "P4"]
USE_CHANNELS = ["O1", "O2", "Oz", "PO3"]
FS_ORIG = 1000
FS_TARGET = 256
T_OFFSET_S = 0.5  # 固定裁掉前0.5s，统一与 eTRCA 分析口径

# 预处理参数
BANDPASS_LOW = 6.0
BANDPASS_HIGH = 100.0
BANDPASS_ORDER = 4
NOTCH_FREQS = [50.0, 100.0]
NOTCH_Q = 35.0

# FBCCA 参数
FBCCA_SRATE = 256
FBCCA_WP = [(6.0, 90.0), (14.0, 90.0), (22.0, 90.0)]
FBCCA_WS = [(4.0, 100.0), (12.0, 100.0), (20.0, 100.0)]
# FBCCA_WP = [(6.0, 50.0), (14.0, 50.0), (22.0, 50.0)]
# FBCCA_WS = [(4.0, 52.0), (12.0, 52.0), (20.0, 52.0)]
FBCCA_FILTER_ORDER = 4
FBCCA_RP = 0.5
FBCCA_N_COMPONENTS = 1
FBCCA_N_HARMONICS = 5


def detect_bad_channels(x_raw, y):
    """坏道检测：零/极低方差、与其他通道相关性极低。"""
    ch_var = {ch: [] for ch in ALL_CHANNELS}
    for trial in x_raw:
        for ch_i, ch_name in enumerate(ALL_CHANNELS):
            ch_var[ch_name].append(np.var(trial[ch_i].astype(np.float64)))

    var_means = {ch: np.mean(ch_var[ch]) for ch in ALL_CHANNELS}
    var_global = np.mean(list(var_means.values()))
    var_std = np.std(list(var_means.values()))

    bad = []
    for ch in ALL_CHANNELS:
        if var_means[ch] < 1e-6:
            bad.append(ch)
            continue
        z = (var_means[ch] - var_global) / (var_std + 1e-12)
        if abs(z) > 2.0:
            bad.append(ch)

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
    data_path = "data/central_controller_ssvep_node4_test/ssvep4_pretrain_dataset_20260409_162235.npy"
    d = np.load(data_path, allow_pickle=True).item()
    x_raw = d["x"]
    y = d["y"]

    # 坏道检测
    bad = detect_bad_channels(x_raw, y)
    print(f"坏道: {bad if bad else '无'}")

    # 只保留指定通道（排除坏道）
    use_channels = [ch for ch in USE_CHANNELS if ch not in bad]
    use_indices = [ALL_CHANNELS.index(ch) for ch in use_channels]
    print(f"使用通道: {use_channels}")
    print(f"固定时间裁剪: 去掉前 {T_OFFSET_S:.2f}s")

    offset_samples = int(round(T_OFFSET_S * FS_TARGET))

    epochs = []
    for trial in x_raw:
        pp = preprocess(trial, use_indices)
        rs = resample_epoch(pp)
        if offset_samples < rs.shape[1]:
            rs = rs[:, offset_samples:]
        epochs.append(rs)

    min_len = min(e.shape[1] for e in epochs)
    epochs = [e[:, :min_len] for e in epochs]

    X = np.stack(epochs, axis=0).astype(np.float64)
    y = np.array(y, dtype=np.int32)
    return X, y, use_channels


def loo_evaluate(X, y, n_harmonics=5):
    """标准 LOO。"""
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
    print("标准 FBCCA LOO 分析（固定裁剪版）")
    print("=" * 80)
    print(f"预处理: 坏道检测 → 去均值 → 去趋势 → 带通{BANDPASS_LOW}-{BANDPASS_HIGH}Hz → "
          f"陷波{NOTCH_FREQS}Hz → 固定裁掉前{T_OFFSET_S}s → 降采样{FS_TARGET}Hz")
    print(f"FBCCA: n_harmonics={FBCCA_N_HARMONICS}")
    print(f"使用通道候选: {USE_CHANNELS}")
    print()

    # ─── 数据准备 ───
    X, y, use_channels = prepare_data()
    print(f"  X.shape={X.shape}, y unique={np.unique(y).tolist()}")
    print()

    # ─── 固定裁剪后的标准 LOO ───
    t0 = time.time()
    acc, per_label = loo_evaluate(X, y, n_harmonics=FBCCA_N_HARMONICS)
    elapsed = time.time() - t0

    print("=" * 80)
    print("汇总")
    print("=" * 80)
    print(f"总准确率: {acc:.4f}  耗时: {elapsed:.1f}s")
    print("各 Label 准确率:")
    for label in sorted(per_label.keys()):
        p = per_label[label]
        la = p["correct"] / p["total"]
        freq = LABEL2FREQ[label]
        print(f"  Label {label} ({freq:>5.1f}Hz): {p['correct']}/{p['total']} = {la:.4f}")
    print()
    print(f"最终使用通道: {use_channels}")


if __name__ == "__main__":
    main()
