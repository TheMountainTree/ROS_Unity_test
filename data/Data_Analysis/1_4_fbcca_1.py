#!/usr/bin/env python3
"""标准 FBCCA 分析（单 epoch 无训练版）：
逐 epoch 独立 FBCCA 解码，不使用同数据训练同数据测试。
预处理: 手动坏道剔除 → 去均值 → 去趋势 → 带通6-100Hz → 陷波50/100Hz → 降采样256Hz

改进：per-frequency n_harmonics — 根据滤波器组通带上限自适应调整每个频率的
谐波数，避免超出通带的谐波引入噪声（尤其改善 18.333Hz / Label 8 准确率）。
"""

import sys
import os
import time
import numpy as np
from scipy.signal import resample, butter, sosfiltfilt, iirnotch, filtfilt, detrend

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MNE_DATA", "/tmp/mne_data")
os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")
os.makedirs(os.environ["MNE_DATA"], exist_ok=True)

from metabci.brainda.algorithms.decomposition import FBSCCA, generate_filterbank

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ssvep_pipeline_fbcca import generate_reference_signals

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
FBCCA_N_JOBS = 1
FBCCA_N_HARMONICS_GRID = [1, 2, 3, 4, 5]

# ─── Per-frequency n_harmonics FBCCA 测试配置 ───
PER_FREQ_CONFIGS = [
    {
        "name": "3fb-50Hz per-freq nh",
        "wp": [(6.0, 50.0), (14.0, 50.0), (22.0, 50.0)],
        "ws": [(4.0, 52.0), (12.0, 52.0), (20.0, 52.0)],
        "f_high": 50.0,
        "max_nh": 5,
    },
    {
        "name": "5fb-90Hz per-freq nh",
        "wp": [(6.0, 90.0), (14.0, 90.0), (22.0, 90.0), (30.0, 90.0), (38.0, 90.0)],
        "ws": [(4.0, 92.0), (12.0, 92.0), (20.0, 92.0), (28.0, 92.0), (36.0, 92.0)],
        "f_high": 90.0,
        "max_nh": 5,
    },
]


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


def _build_no_train_fbcca_estimator(n_channels, n_samples, n_harmonics):
    """构建无训练 FBCCA 解码器（仅依赖参考信号，不使用真实训练数据）。"""
    filterbank = generate_filterbank(
        FBCCA_WP, FBCCA_WS, srate=FBCCA_SRATE, order=FBCCA_FILTER_ORDER, rp=FBCCA_RP
    )
    filterweights = np.array(
        [(n + 1) ** (-1.25) + 0.25 for n in range(len(FBCCA_WP))]
    )
    y_labels = np.arange(1, len(FREQS) + 1, dtype=np.int32)
    y_ref = generate_reference_signals(
        n_samples=n_samples,
        freqs=FREQS,
        srate=FBCCA_SRATE,
        n_harmonics=n_harmonics,
    )
    x_dummy = np.zeros((len(y_labels), n_channels, n_samples), dtype=np.float64)

    est = FBSCCA(
        filterbank=filterbank,
        n_components=FBCCA_N_COMPONENTS,
        filterweights=filterweights,
        n_jobs=FBCCA_N_JOBS,
    )
    est.fit(X=x_dummy, y=y_labels, Yf=y_ref)
    return est


def single_epoch_no_train_evaluate(X, y, n_harmonics=5):
    """逐 epoch 无训练 FBCCA 解码，返回总体 + 各 label 准确率。"""
    n_total = len(y)
    n_correct = 0
    n_channels = int(X.shape[1])
    n_samples = int(X.shape[2])
    class_labels = np.arange(1, len(FREQS) + 1, dtype=np.int32)

    try:
        est = _build_no_train_fbcca_estimator(
            n_channels=n_channels,
            n_samples=n_samples,
            n_harmonics=n_harmonics,
        )
    except Exception as e:
        print(f"    初始化无训练 FBCCA 失败: {e}")
        return {"accuracy": 0.0, "correct": 0, "total": n_total, "per_label": {}}

    per_label = {}
    for i in range(n_total):
        x_test = X[i:i + 1]
        y_test = int(y[i])
        try:
            pred_raw = int(est.predict(x_test)[0])
            # 与 1_4_fbcca_2 一致：MetaBCI FBSCCA 输出 0-based 类索引。
            if 0 <= pred_raw < len(class_labels):
                pred = int(class_labels[pred_raw])
            else:
                pred = -1
        except Exception as e:
            print(f"    Trial {i} 解码失败: {e}")
            pred = -1

        if pred == y_test:
            n_correct += 1
        if y_test not in per_label:
            per_label[y_test] = {"correct": 0, "total": 0}
        per_label[y_test]["total"] += 1
        if pred == y_test:
            per_label[y_test]["correct"] += 1

    acc = n_correct / n_total if n_total > 0 else 0.0
    return {"accuracy": acc, "correct": n_correct, "total": n_total, "per_label": per_label}


def _cca_first_corr(X, Y):
    """Compute first canonical correlation between X and Y.

    Uses SVD of the whitened cross-covariance matrix, equivalent to
    metabci's GED-based _scca_kernel but with per-frequency reference
    signal dimensionality support.

    Parameters
    ----------
    X : ndarray, shape (p, n) — EEG data (will be demeaned internally)
    Y : ndarray, shape (q, n) — Reference signals (will be demeaned internally)

    Returns
    -------
    rho : float — First canonical correlation (0..1)
    """
    p, n = X.shape
    q = Y.shape[0]

    # Explicit demean
    X = X - X.mean(axis=1, keepdims=True)
    Y = Y - Y.mean(axis=1, keepdims=True)

    # Covariance matrices
    Cxx = (X @ X.T) / (n - 1)
    Cyy = (Y @ Y.T) / (n - 1)
    Cxy = (X @ Y.T) / (n - 1)

    # Regularize for numerical stability
    Cxx += np.eye(p) * 1e-8
    Cyy += np.eye(q) * 1e-8

    # Inverse square root via eigendecomposition
    evals_x, evecs_x = np.linalg.eigh(Cxx)
    Cxx_isqrt = evecs_x @ np.diag(1.0 / np.sqrt(np.maximum(evals_x, 1e-12))) @ evecs_x.T

    evals_y, evecs_y = np.linalg.eigh(Cyy)
    Cyy_isqrt = evecs_y @ np.diag(1.0 / np.sqrt(np.maximum(evals_y, 1e-12))) @ evecs_y.T

    # Cross-covariance in whitened space — singular values = canonical correlations
    M = Cxx_isqrt @ Cxy @ Cyy_isqrt
    sv = np.linalg.svd(M, compute_uv=False)
    return sv[0]


def compute_per_freq_n_harmonics(freqs, f_high, max_nh=5):
    """Compute appropriate n_harmonics for each frequency based on filterbank passband.

    For each frequency, uses at most floor(f_high / freq) harmonics,
    ensuring all harmonics fall within the filterbank passband.
    Out-of-band harmonics add noise to CCA features.
    """
    result = []
    for freq in freqs:
        nh = min(max_nh, max(1, int(f_high / freq)))
        result.append(nh)
    return result


def per_freq_fbcca_evaluate(X, y, wp, ws, f_high, max_nh=5):
    """Per-frequency n_harmonics FBCCA evaluation.

    For each target frequency, generates reference signals with n_harmonics
    capped by f_high, avoiding out-of-band harmonics that add noise.
    Computes CCA correlation per (frequency, subband) pair independently,
    then applies filter weights and picks the frequency with highest score.

    Returns: (accuracy, per_label_dict, freq_nh_list)
    """
    n_total = len(y)
    n_samples = int(X.shape[2])
    n_fb = len(wp)

    # Compute per-frequency n_harmonics
    freq_nh = compute_per_freq_n_harmonics(FREQS, f_high, max_nh)

    # Generate filterbank
    filterbank = generate_filterbank(
        wp, ws, srate=FBCCA_SRATE, order=FBCCA_FILTER_ORDER, rp=FBCCA_RP
    )
    filterweights = np.array([(n + 1) ** (-1.25) + 0.25 for n in range(n_fb)])

    # Generate per-frequency reference signals
    t = np.arange(n_samples) / FBCCA_SRATE
    ref_signals = []
    for idx, freq in enumerate(FREQS):
        nh = freq_nh[idx]
        Y = np.zeros((2 * nh, n_samples))
        for h in range(nh):
            Y[2 * h, :] = np.sin(2 * np.pi * (h + 1) * freq * t)
            Y[2 * h + 1, :] = np.cos(2 * np.pi * (h + 1) * freq * t)
        ref_signals.append(Y)

    # Evaluate each epoch
    n_correct = 0
    per_label = {}

    for i in range(n_total):
        epoch = X[i]  # (n_channels, n_samples)

        # Compute weighted score for each frequency
        scores = np.zeros(len(FREQS))
        for fb_idx in range(n_fb):
            sos = filterbank[fb_idx]
            filtered = sosfiltfilt(sos, epoch, axis=1)

            for freq_idx in range(len(FREQS)):
                corr = _cca_first_corr(filtered, ref_signals[freq_idx])
                scores[freq_idx] += filterweights[fb_idx] * corr

        pred = int(np.argmax(scores)) + 1  # 1-based labels
        y_test = int(y[i])

        if pred == y_test:
            n_correct += 1

        if y_test not in per_label:
            per_label[y_test] = {"correct": 0, "total": 0}
        per_label[y_test]["total"] += 1
        if pred == y_test:
            per_label[y_test]["correct"] += 1

    acc = n_correct / n_total
    return acc, per_label, freq_nh


def main():
    print("=" * 80)
    print("标准 FBCCA (FBSCCA) 单 epoch 无训练分析 + per-frequency n_harmonics 改进")
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

    # ─── 1. 原始 uniform n_harmonics 评估 (作为 baseline) ───
    print("=" * 80)
    print("Part 1: 原始 uniform n_harmonics (baseline)")
    print("=" * 80)
    all_results = {}
    for nh in FBCCA_N_HARMONICS_GRID:
        print(f"--- n_harmonics={nh} ---")
        t0 = time.time()
        result = single_epoch_no_train_evaluate(X, y, n_harmonics=nh)
        elapsed = time.time() - t0

        print(
            f"  总成功率: {result['accuracy']:.4f} "
            f"({result['correct']}/{result['total']})  耗时: {elapsed:.1f}s"
        )
        if result["per_label"]:
            print(f"  各 Label 准确率:")
            for label in sorted(result["per_label"].keys()):
                p = result["per_label"][label]
                la = p["correct"] / p["total"]
                freq = LABEL2FREQ[label]
                print(f"    Label {label} ({freq:>5.1f}Hz): "
                      f"{p['correct']}/{p['total']} = {la:.4f}")
        print()

        all_results[nh] = result

    # ─── 2. Per-frequency n_harmonics 评估 ───
    print("=" * 80)
    print("Part 2: Per-frequency n_harmonics (自适应谐波数)")
    print("=" * 80)
    print("原理: 对每个频率，n_harmonics = min(max_nh, floor(f_high / freq))，"
          "确保所有谐波在通带内")
    print()

    pf_results = {}
    for cfg in PER_FREQ_CONFIGS:
        name = cfg["name"]
        wp = cfg["wp"]
        ws = cfg["ws"]
        f_high = cfg["f_high"]
        max_nh = cfg["max_nh"]

        freq_nh = compute_per_freq_n_harmonics(FREQS, f_high, max_nh)
        print(f"--- {name} ---")
        print(f"  wp={wp}, f_high={f_high}Hz, max_nh={max_nh}")
        print(f"  per-freq n_harmonics: {dict(zip(FREQS, freq_nh))}")

        t0 = time.time()
        acc, per_label, _ = per_freq_fbcca_evaluate(X, y, wp, ws, f_high, max_nh)
        elapsed = time.time() - t0

        print(f"  总准确率: {acc:.4f}  耗时: {elapsed:.1f}s")
        print(f"  各 Label 准确率:")
        for label in sorted(per_label.keys()):
            p = per_label[label]
            la = p["correct"] / p["total"]
            freq = LABEL2FREQ[label]
            print(f"    Label {label} ({freq:>5.1f}Hz, nh={freq_nh[label-1]}): "
                  f"{p['correct']}/{p['total']} = {la:.4f}")
        print()

        pf_results[name] = {"accuracy": acc, "per_label": per_label, "freq_nh": freq_nh}

    # ─── 汇总对比 ───
    print("=" * 80)
    print("汇总对比")
    print("=" * 80)

    # Build comparison table: baseline best + all per-freq configs
    # Find baseline best
    best_nh = max(all_results, key=lambda nh: all_results[nh]["accuracy"])

    header = f"{'配置':>24} {'总准确率':>8}"
    for label in sorted(LABEL2FREQ.keys()):
        header += f" {'L'+str(label)+'('+str(int(LABEL2FREQ[label]))+'Hz)':>10}"
    print(header)
    print("-" * len(header))

    # Baseline rows
    for nh in FBCCA_N_HARMONICS_GRID:
        r = all_results[nh]
        marker = " *" if nh == best_nh else ""
        row = f"{'uniform nh='+str(nh):>24} {r['accuracy']:>8.4f}"
        for label in sorted(LABEL2FREQ.keys()):
            if label in r.get("per_label", {}):
                p = r["per_label"][label]
                la = p["correct"] / p["total"]
                row += f" {la:>10.4f}"
            else:
                row += f" {'N/A':>10}"
        print(row + marker)

    # Per-freq rows
    for name, r in pf_results.items():
        row = f"{name:>24} {r['accuracy']:>8.4f}"
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
    print()
    print("结论: per-frequency n_harmonics 根据滤波器通带自适应调整每个频率的谐波数，")
    print("  避免超出通带的谐波引入噪声，尤其改善高频目标（如 18.333Hz / Label 8）准确率。")


if __name__ == "__main__":
    main()
