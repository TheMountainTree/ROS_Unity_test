#!/usr/bin/env python3
"""eTRCA 分析：对 ssvep4_pretrain_dataset 使用 4 trials 训练，1 trial 测试

数据: 5 trials × 8 labels, shape per epoch = (8 channels, ~2020 samples @ 1000Hz)
方法: FBTRCA (ensemble TRCA) + filterbank
预处理: 去均值 → 去趋势 → 高通6Hz → 陷波50/100Hz → 降采样256Hz
训练: 前 4 trials/label (32 epochs)
测试: 第 5 trial/label (8 epochs)
"""

import sys
import os
import time
import numpy as np
from scipy.signal import resample, butter, sosfiltfilt, iirnotch, filtfilt, detrend

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ssvep_pipeline import SSVEPPretrainer, SSVEPDecoder

# ─── 全局参数 ───
FREQS = [8.684, 9.706, 11.0, 11.786, 12.692, 13.75, 15.0, 18.333]
LABEL2FREQ = {i + 1: f for i, f in enumerate(FREQS)}
ALL_CHANNELS = ["O1", "O2", "Oz", "PO3", "PO4", "Pz", "P3", "P4"]
FS_ORIG = 1000
FS_TARGET = 256

# 预处理参数
HIGHPASS_CUTOFF = 6.0
HIGHPASS_ORDER = 4
NOTCH_FREQS = [50.0, 100.0]
NOTCH_Q = 35.0

# eTRCA 参数
ETRCA_SRATE = 256
ETRCA_WP = [(6, 50), (14, 50), (22, 50)]
ETRCA_WS = [(4, 52), (12, 52), (20, 52)]
ETRCA_FILTER_ORDER = 4
ETRCA_RP = 0.5
ETRCA_N_COMPONENTS_GRID = [1]
ETRCA_ENSEMBLE = True


def preprocess(trial):
    """对单个 trial 做预处理，保留所有通道。

    流程: 去均值 → 去趋势 → 高通 → 陷波 → 降采样
    """
    out = trial.astype(np.float64, copy=True)
    # 去均值
    out -= np.mean(out, axis=1, keepdims=True)
    # 去趋势
    out = detrend(out, axis=1)
    # 高通
    nyq = FS_ORIG * 0.5
    if HIGHPASS_CUTOFF < nyq:
        sos_hp = butter(
            HIGHPASS_ORDER, HIGHPASS_CUTOFF, btype="highpass", fs=FS_ORIG, output="sos"
        )
        out = sosfiltfilt(sos_hp, out, axis=1)
    # 陷波
    for f0 in NOTCH_FREQS:
        if f0 < FS_ORIG / 2:
            b, a = iirnotch(w0=f0, Q=NOTCH_Q, fs=FS_ORIG)
            out = filtfilt(b, a, out, axis=1)
    return out


def resample_epoch(epoch):
    n_target = int(epoch.shape[1] * FS_TARGET / FS_ORIG)
    return resample(epoch, n_target, axis=1)


def prepare_data(data_path):
    """加载数据，预处理，按 label 分组返回 train/test split。

    每个 label 有 5 trials，取前 4 个训练，第 5 个测试。
    返回 X_train, y_train, X_test, y_test
    """
    d = np.load(data_path, allow_pickle=True).item()
    x_raw = d["x"]
    y = d["y"]

    # 预处理 + 降采样
    epochs = []
    for trial in x_raw:
        pp = preprocess(trial)
        rs = resample_epoch(pp)
        epochs.append(rs)

    # 对齐长度
    min_len = min(e.shape[1] for e in epochs)
    epochs = [e[:, :min_len] for e in epochs]

    X_all = np.stack(epochs, axis=0).astype(np.float64)
    y_all = np.array(y, dtype=np.int32)

    # 按 label 分组，每个 label 的 trial 按原始顺序排列
    train_indices = []
    test_indices = []
    for label in sorted(np.unique(y_all)):
        label_indices = np.where(y_all == label)[0]
        # 每个 label 有 5 trials，取前 4 个训练，最后一个测试
        train_indices.extend(label_indices[:4])
        test_indices.append(label_indices[4])

    X_train = X_all[train_indices]
    y_train = y_all[train_indices]
    X_test = X_all[test_indices]
    y_test = y_all[test_indices]

    return X_train, y_train, X_test, y_test


def train_and_evaluate(X_train, y_train, X_test, y_test, n_components=1):
    """训练 eTRCA 模型并在测试集上评估。"""
    pretrainer = SSVEPPretrainer(
        srate=ETRCA_SRATE,
        wp=ETRCA_WP,
        ws=ETRCA_WS,
        filter_order=ETRCA_FILTER_ORDER,
        rp=ETRCA_RP,
        n_components=n_components,
        ensemble=ETRCA_ENSEMBLE,
    )
    pretrainer.fit(X_train, y_train)

    decoder = SSVEPDecoder(pretrainer)
    y_pred = decoder.decode(X_test)

    # eTRCA (FBTRCA with ensemble=True) predict returns training labels directly
    # 不需要像 FBCCA 那样做 index → label 映射
    n_correct = int(np.sum(y_pred == y_test))
    accuracy = n_correct / len(y_test)

    per_label = {}
    for i, (true, pred) in enumerate(zip(y_test, y_pred)):
        label = int(true)
        if label not in per_label:
            per_label[label] = {"correct": 0, "total": 0, "predictions": []}
        per_label[label]["total"] += 1
        per_label[label]["predictions"].append(int(pred))
        if pred == true:
            per_label[label]["correct"] += 1

    return accuracy, per_label, y_pred


def loo_evaluate(X_all, y_all, n_components=1):
    """LOO 交叉验证（在全部 40 trials 上）"""
    n = len(y_all)
    correct = 0
    per_label = {}

    for i in range(n):
        X_train = np.delete(X_all, i, axis=0)
        y_train = np.delete(y_all, i)
        X_test = X_all[i:i + 1]
        y_test = y_all[i]

        try:
            pretrainer = SSVEPPretrainer(
                srate=ETRCA_SRATE,
                wp=ETRCA_WP,
                ws=ETRCA_WS,
                filter_order=ETRCA_FILTER_ORDER,
                rp=ETRCA_RP,
                n_components=n_components,
                ensemble=ETRCA_ENSEMBLE,
            )
            pretrainer.fit(X_train, y_train)
            decoder = SSVEPDecoder(pretrainer)
            pred = int(decoder.decode(X_test)[0])
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
    data_path = "data/central_controller_ssvep_node4_test/ssvep4_pretrain_dataset_20260416_164519.npy"

    print("=" * 80)
    print("eTRCA (FBTRCA) 分析 — 4 trials 训练 + 1 trial 测试")
    print("=" * 80)
    print(f"数据文件: {data_path}")
    print(f"预处理: 去均值 → 去趋势 → 高通{HIGHPASS_CUTOFF}Hz → 陷波{NOTCH_FREQS}Hz → 降采样{FS_TARGET}Hz")
    print(f"eTRCA: srate={ETRCA_SRATE}, wp={ETRCA_WP}, ensemble={ETRCA_ENSEMBLE}")
    print(f"目标频率: {FREQS}")
    print(f"通道: {ALL_CHANNELS}")
    print()

    # ─── 数据加载与分割 ───
    X_train, y_train, X_test, y_test = prepare_data(data_path)
    print(f"训练集: X.shape={X_train.shape}, labels={np.unique(y_train).tolist()}")
    print(f"  每个 label 的 trial 数: {sum(y_train == 1)}")
    print(f"测试集: X.shape={X_test.shape}, y_test={y_test.tolist()}")
    print()

    # ─── 4-train / 1-test 评估 ───
    print("=" * 80)
    print("1. 4 trials 训练 + 1 trial 测试")
    print("=" * 80)

    for nc in ETRCA_N_COMPONENTS_GRID:
        t0 = time.time()
        acc, per_label, y_pred = train_and_evaluate(X_train, y_train, X_test, y_test, n_components=nc)
        elapsed = time.time() - t0

        print(f"\n--- n_components={nc}, ensemble={ETRCA_ENSEMBLE} ---")
        print(f"  测试准确率: {acc:.4f} ({int(acc * len(y_test))}/{len(y_test)})")
        print(f"  耗时: {elapsed:.2f}s")
        print(f"  逐 label 结果:")
        for label in sorted(per_label.keys()):
            p = per_label[label]
            la = p["correct"] / p["total"] if p["total"] > 0 else 0.0
            freq = LABEL2FREQ[label]
            preds = p["predictions"]
            print(f"    Label {label} ({freq:>7.3f}Hz): {p['correct']}/{p['total']} = {la:.4f}  "
                  f"pred={preds}")

    # ─── LOO 交叉验证（全 40 trials） ───
    print()
    print("=" * 80)
    print("2. LOO 交叉验证 (全 40 trials)")
    print("=" * 80)

    # 重新加载全量数据
    d = np.load(data_path, allow_pickle=True).item()
    x_raw = d["x"]
    y_raw = d["y"]
    epochs = []
    for trial in x_raw:
        pp = preprocess(trial)
        rs = resample_epoch(pp)
        epochs.append(rs)
    min_len = min(e.shape[1] for e in epochs)
    epochs = [e[:, :min_len] for e in epochs]
    X_all = np.stack(epochs, axis=0).astype(np.float64)
    y_all = np.array(y_raw, dtype=np.int32)

    for nc in ETRCA_N_COMPONENTS_GRID:
        print(f"\n--- n_components={nc}, ensemble={ETRCA_ENSEMBLE} ---")
        t0 = time.time()
        acc, per_label = loo_evaluate(X_all, y_all, n_components=nc)
        elapsed = time.time() - t0
        print(f"  LOO 准确率: {acc:.4f}  耗时: {elapsed:.1f}s")
        print(f"  各 Label 准确率:")
        for label in sorted(per_label.keys()):
            p = per_label[label]
            la = p["correct"] / p["total"]
            freq = LABEL2FREQ[label]
            print(f"    Label {label} ({freq:>7.3f}Hz): {p['correct']}/{p['total']} = {la:.4f}")

    # ─── 汇总对比 ───
    print()
    print("=" * 80)
    print("3. 汇总")
    print("=" * 80)
    print(f"  训练集: 每个 label 4 trials, 共 32 trials")
    print(f"  测试集: 每个 label 1 trial,  共 8 trials")
    print(f"  算法:   FBTRCA (eTRCA), ensemble={ETRCA_ENSEMBLE}")
    print(f"  预处理: 高通{HIGHPASS_CUTOFF}Hz + 陷波{NOTCH_FREQS}Hz + 降采样{FS_TARGET}Hz")
    print(f"  Filterbank: wp={ETRCA_WP}")


if __name__ == "__main__":
    main()
