#!/usr/bin/env python3
"""标准 eTRCA 留一分析（合并多 pretrain 数据集）：
自动加载 central_controller_ssvep_node4_test 下所有 pretrain dataset，
合并后执行 leave-one-trial-out（每次留1个 trial 测试，其余用于训练）。
训练阶段加入随机时间扰动增强；测试阶段使用多偏移投票解码。

预处理: 坏道检测 → 去均值 → 去趋势 → 带通6-100Hz → 陷波50/100Hz → 截取0.5s后 → 降采样256Hz
"""

import sys
import os
import time
import glob
import numpy as np
from scipy.signal import resample, butter, sosfiltfilt, iirnotch, filtfilt, detrend

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ssvep_pipeline import SSVEPPretrainer, SSVEPDecoder

# ─── 全局参数 ───
FREQS = [8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 40.0, 45.0]
LABEL2FREQ = {i + 1: f for i, f in enumerate(FREQS)}
ALL_CHANNELS = ["O1", "O2", "Oz", "PO3", "PO4", "Pz", "P3", "P4"]
FS_ORIG = 1000
FS_TARGET = 256

# 预处理参数
BANDPASS_LOW = 6.0
BANDPASS_HIGH = 100.0
BANDPASS_ORDER = 4
NOTCH_FREQS = [50.0, 100.0]
NOTCH_Q = 35.0
ROBUST_CHANNEL_NORM = True
T_OFFSET_S = 0.5  # 截掉前0.5s，去除刺激 onset 瞬态

# eTRCA 参数
ETRCA_SRATE = 256
ETRCA_WP = [(6.0, 100.0), (14.0, 100.0), (22.0, 100.0)]
ETRCA_WS = [(4.0, 102.0), (12.0, 102.0), (20.0, 102.0)]
ETRCA_FILTER_ORDER = 4
ETRCA_RP = 0.5
ETRCA_N_COMPONENTS_GRID = [1, 2, 3]
ETRCA_ENSEMBLE = True

# 训练扰动增强
TRAIN_AUG_ENABLED = True
TRAIN_AUG_COPIES = 2
TRAIN_JITTER_MAX_SAMPLES = 4

# 测试多偏移投票（单位: sample @256Hz）
TEST_VOTE_OFFSETS = [-4, -2, 0, 2, 4]
RNG_SEED = 42
PRETRAIN_DATASET_GLOB = "data/central_controller_ssvep_node4_test/ssvep4_pretrain_dataset_*.npy"


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
    """预处理，只保留 good_indices 对应的通道。"""
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

    if ROBUST_CHANNEL_NORM and out.ndim == 2:
        centered = out - np.median(out, axis=1, keepdims=True)
        mad = np.median(np.abs(centered), axis=1)
        scale = 1.4826 * mad
        valid = scale > 1e-12
        global_scale = float(np.median(scale[valid])) if np.any(valid) else 1.0
        floor = max(1e-6, global_scale * 0.05)
        out = centered / np.maximum(scale, floor)[:, np.newaxis]

    return out


def resample_epoch(epoch, target_fs=FS_TARGET):
    n_target = int(epoch.shape[1] * target_fs / FS_ORIG)
    return resample(epoch, n_target, axis=1)


def prepare_data():
    """加载并合并多 pretrain 数据集，检测坏道，预处理。"""
    data_paths = sorted(glob.glob(PRETRAIN_DATASET_GLOB))
    if not data_paths:
        raise FileNotFoundError(f"未找到 pretrain 数据集: {PRETRAIN_DATASET_GLOB}")

    x_raw_all = []
    y_all = []
    used_paths = []
    skipped_paths = []
    for path in data_paths:
        d = np.load(path, allow_pickle=True).item()
        x_raw = d.get("x", [])
        y = d.get("y", [])
        if len(x_raw) == 0 or len(y) == 0:
            skipped_paths.append((path, len(y)))
            continue
        x_raw_all.extend(list(x_raw))
        y_all.extend(list(y))
        used_paths.append((path, len(y)))

    if len(y_all) == 0:
        raise RuntimeError("所有 pretrain 数据集均为空，无法训练。")

    x_raw = x_raw_all
    y = np.array(y_all, dtype=np.int32)

    print("加载的 pretrain dataset:")
    for path, n_trials in used_paths:
        print(f"  + {os.path.basename(path)}: {n_trials} trials")
    for path, n_trials in skipped_paths:
        print(f"  - {os.path.basename(path)}: {n_trials} trials (跳过空数据)")

    # 坏道检测（在原始数据上，先于任何预处理）
    bad = detect_bad_channels(x_raw, y)
    good_channels = [ch for ch in ALL_CHANNELS if ch not in bad]
    good_indices = [ALL_CHANNELS.index(ch) for ch in good_channels]

    print(f"坏道: {bad if bad else '无'}")
    print(f"使用通道: {good_channels}")

    offset_samples = int(round(T_OFFSET_S * FS_TARGET))

    epochs = []
    for trial in x_raw:
        pp = preprocess(trial, good_indices)
        rs = resample_epoch(pp)
        if offset_samples < rs.shape[1]:
            rs = rs[:, offset_samples:]
        epochs.append(rs)

    min_len = min(e.shape[1] for e in epochs)
    epochs = [e[:, :min_len] for e in epochs]

    X = np.stack(epochs, axis=0).astype(np.float64)
    return X, y, good_channels


def shift_epoch_with_zeros(epoch: np.ndarray, shift: int) -> np.ndarray:
    """Time shift epoch with zero padding (no circular wrap)."""
    out = np.zeros_like(epoch)
    if shift == 0:
        return epoch.copy()
    if shift > 0:
        out[:, shift:] = epoch[:, :-shift]
    else:
        s = -shift
        out[:, :-s] = epoch[:, s:]
    return out


def augment_training_data(X_train: np.ndarray, y_train: np.ndarray, rng: np.random.Generator):
    """随机时间扰动增强：原样本 + 随机平移副本。"""
    if (not TRAIN_AUG_ENABLED) or TRAIN_AUG_COPIES <= 0 or TRAIN_JITTER_MAX_SAMPLES <= 0:
        return X_train, y_train

    x_list = [X_train]
    y_list = [y_train]
    for _ in range(TRAIN_AUG_COPIES):
        shifts = rng.integers(
            low=-TRAIN_JITTER_MAX_SAMPLES,
            high=TRAIN_JITTER_MAX_SAMPLES + 1,
            size=X_train.shape[0],
        )
        x_aug = np.stack(
            [shift_epoch_with_zeros(X_train[i], int(shifts[i])) for i in range(X_train.shape[0])],
            axis=0,
        )
        x_list.append(x_aug)
        y_list.append(y_train.copy())

    return np.concatenate(x_list, axis=0), np.concatenate(y_list, axis=0)


def vote_decode_epoch(dec: SSVEPDecoder, epoch: np.ndarray, offsets):
    """对单个 epoch 做多偏移投票解码。"""
    votes = {}
    for shift in offsets:
        x_shift = shift_epoch_with_zeros(epoch, int(shift))[np.newaxis, :, :]
        pred = int(dec.decode(x_shift)[0])
        votes[pred] = votes.get(pred, 0) + 1
    # 票数优先，标签值次优（稳定可复现）
    return sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def leave_one_trial_out_evaluate(X, y, n_components=1):
    """全局留一：每次留1个 trial 测试，其余用于训练。"""
    rng = np.random.default_rng(RNG_SEED)
    n_folds = len(y)
    if n_folds < 2:
        raise ValueError("trial 数不足，无法做留一评估。")
    correct = 0
    per_label = {}
    n_total = n_folds

    for fold in range(n_folds):
        train_mask = np.ones(n_folds, dtype=bool)
        train_mask[fold] = False

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[fold]
        y_test = int(y[fold])

        X_train_aug, y_train_aug = augment_training_data(X_train, y_train, rng)

        try:
            pt = SSVEPPretrainer(
                srate=ETRCA_SRATE, wp=ETRCA_WP, ws=ETRCA_WS,
                filter_order=ETRCA_FILTER_ORDER, rp=ETRCA_RP,
                n_components=n_components, ensemble=ETRCA_ENSEMBLE,
            )
            pt.fit(X_train_aug, y_train_aug)
            dec = SSVEPDecoder(pt)
        except Exception as e:
            print(f"    Fold {fold}: error {e}")
            dec = None

        if dec is None:
            pred = -1
        else:
            pred = vote_decode_epoch(dec, X_test, TEST_VOTE_OFFSETS)

        if y_test not in per_label:
            per_label[y_test] = {"correct": 0, "total": 0}
        per_label[y_test]["total"] += 1

        if pred == y_test:
            correct += 1
            per_label[y_test]["correct"] += 1

        if (fold + 1) % 10 == 0 or (fold + 1) == n_folds:
            print(f"    Fold {fold + 1}/{n_folds}: 当前累计 {correct}/{fold + 1} = {correct/(fold + 1):.4f}")

    acc = correct / n_total if n_total > 0 else 0.0
    return acc, per_label


def main():
    print("=" * 80)
    print("标准 eTRCA 留一分析（合并多 pretrain 数据集）")
    print("=" * 80)
    print(f"预处理: 坏道检测 → 去均值 → 去趋势 → 带通{BANDPASS_LOW}-{BANDPASS_HIGH}Hz → "
          f"陷波{NOTCH_FREQS}Hz → 截取{T_OFFSET_S}s后 → 降采样{FS_TARGET}Hz")
    print(f"eTRCA: srate={ETRCA_SRATE}, wp={ETRCA_WP}, "
          f"n_components_grid={ETRCA_N_COMPONENTS_GRID}, ensemble={ETRCA_ENSEMBLE}")
    print(f"训练增强: enabled={TRAIN_AUG_ENABLED}, copies={TRAIN_AUG_COPIES}, jitter=±{TRAIN_JITTER_MAX_SAMPLES} samples")
    print(f"测试投票偏移: {TEST_VOTE_OFFSETS}")
    print(f"目标频率: {FREQS}")
    print(f"数据集匹配: {PRETRAIN_DATASET_GLOB}")
    print()

    # ─── 数据准备 ───
    X, y, good_channels = prepare_data()
    print(f"  X.shape={X.shape}, y unique={np.unique(y).tolist()}")
    print()

    # ─── 留一评估 (n_components 网格) ───
    all_results = {}
    for n_comp in ETRCA_N_COMPONENTS_GRID:
        print(f"--- n_components={n_comp} ---")
        t0 = time.time()
        acc, per_label = leave_one_trial_out_evaluate(X, y, n_components=n_comp)
        elapsed = time.time() - t0

        print(f"  总准确率: {acc:.4f}  耗时: {elapsed:.1f}s")
        print(f"  各 Label 准确率:")
        for label in sorted(per_label.keys()):
            p = per_label[label]
            la = p["correct"] / p["total"]
            freq = LABEL2FREQ[label]
            print(f"    Label {label} ({freq:>5.1f}Hz): {p['correct']}/{p['total']} = {la:.4f}")
        print()

        all_results[n_comp] = {"accuracy": acc, "per_label": per_label}

    # ─── 汇总 ───
    print("=" * 80)
    print("汇总")
    print("=" * 80)

    header = f"{'n_components':>12} {'总准确率':>8}"
    for label in sorted(LABEL2FREQ.keys()):
        header += f" {'L'+str(label)+'('+str(int(LABEL2FREQ[label]))+'Hz)':>10}"
    print(header)
    print("-" * len(header))

    for n_comp in ETRCA_N_COMPONENTS_GRID:
        r = all_results[n_comp]
        row = f"{n_comp:>12} {r['accuracy']:>8.4f}"
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
