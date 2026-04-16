#!/usr/bin/env python3
"""eTRCA Session 报告 + 跨 Session 验证矩阵。

功能:
1. 按 session 分开报告（session 内 leave-one-trial-out）。
2. 跨 session 验证矩阵（训练 session_i -> 测试 session_j）。

说明:
- 自动读取 central_controller_ssvep_node4_test 下所有 pretrain dataset。
- 空数据集自动跳过。
- 预处理、训练增强、测试投票与 1_4_preprocess.py 保持一致口径。
"""

import glob
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import butter, detrend, filtfilt, iirnotch, resample, sosfiltfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ssvep_pipeline import SSVEPDecoder, SSVEPPretrainer

# ─── 全局参数 ───
FREQS = [8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 40.0, 45.0]
LABEL2FREQ = {i + 1: f for i, f in enumerate(FREQS)}
ALL_CHANNELS = ["O1", "O2", "Oz", "PO3", "PO4", "Pz", "P3", "P4"]

FS_ORIG = 1000
FS_TARGET = 256
T_OFFSET_S = 0.5

PRETRAIN_DATASET_GLOB = "data/central_controller_ssvep_node4_test/ssvep4_pretrain_dataset_*.npy"

# 预处理参数
BANDPASS_LOW = 6.0
BANDPASS_HIGH = 100.0
BANDPASS_ORDER = 4
NOTCH_FREQS = [50.0, 100.0]
NOTCH_Q = 35.0
ROBUST_CHANNEL_NORM = True

# eTRCA 参数
ETRCA_SRATE = 256
ETRCA_WP = [(6.0, 100.0), (14.0, 100.0), (22.0, 100.0)]
ETRCA_WS = [(4.0, 102.0), (12.0, 102.0), (20.0, 102.0)]
ETRCA_FILTER_ORDER = 4
ETRCA_RP = 0.5
ETRCA_N_COMPONENTS_GRID = [1, 2, 3]
ETRCA_ENSEMBLE = True

# 训练增强
TRAIN_AUG_ENABLED = True
TRAIN_AUG_COPIES = 2
TRAIN_JITTER_MAX_SAMPLES = 4

# 测试投票偏移
TEST_VOTE_OFFSETS = [-4, -2, 0, 2, 4]
RNG_SEED = 42

# 候选窗口（相对于截掉前 T_OFFSET_S 后的时间轴）
WINDOW_CANDIDATES_S = [
    (0.00, 1.00),
    (0.00, 1.50),
    (0.25, 1.25),
    (0.50, 1.50),
]
MIN_WINDOW_SAMPLES = 128


def detect_bad_channels(x_raw: List[np.ndarray]) -> List[str]:
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
        return sorted(set(bad))

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
        return sorted(set(bad))

    corr_avg = corr_accum / n_valid
    for ii, ch in enumerate(good_names):
        off_diag = np.delete(corr_avg[ii, :], ii)
        if np.mean(off_diag) < 0.3:
            bad.append(ch)

    return sorted(set(bad))


def preprocess_trial(trial: np.ndarray, good_indices: List[int]) -> np.ndarray:
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

    n_target = int(out.shape[1] * FS_TARGET / FS_ORIG)
    out = resample(out, n_target, axis=1)

    offset_samples = int(round(T_OFFSET_S * FS_TARGET))
    if offset_samples < out.shape[1]:
        out = out[:, offset_samples:]
    return out


def load_sessions() -> Tuple[List[Dict], List[str], List[int]]:
    data_paths = sorted(glob.glob(PRETRAIN_DATASET_GLOB))
    if not data_paths:
        raise FileNotFoundError(f"未找到 pretrain 数据集: {PRETRAIN_DATASET_GLOB}")

    raw_sessions = []
    all_trials = []
    for path in data_paths:
        d = np.load(path, allow_pickle=True).item()
        x_raw = list(d.get("x", []))
        y = np.array(d.get("y", []), dtype=np.int32)
        if len(x_raw) == 0 or len(y) == 0:
            print(f"- 跳过空数据: {os.path.basename(path)}")
            continue
        raw_sessions.append({"name": os.path.basename(path), "x_raw": x_raw, "y": y})
        all_trials.extend(x_raw)

    if not raw_sessions:
        raise RuntimeError("所有 pretrain 数据集均为空。")

    bad = detect_bad_channels(all_trials)
    good_channels = [ch for ch in ALL_CHANNELS if ch not in bad]
    good_indices = [ALL_CHANNELS.index(ch) for ch in good_channels]

    print(f"坏道(全局): {bad if bad else '无'}")
    print(f"使用通道(全局): {good_channels}")

    sessions = []
    min_len = None
    for s in raw_sessions:
        epochs = [preprocess_trial(tr, good_indices) for tr in s["x_raw"]]
        this_min = min(ep.shape[1] for ep in epochs)
        min_len = this_min if min_len is None else min(min_len, this_min)
        sessions.append({"name": s["name"], "epochs": epochs, "y": s["y"]})

    # 为了跨 session 可比，统一截到全局最短长度
    for s in sessions:
        X = np.stack([ep[:, :min_len] for ep in s["epochs"]], axis=0).astype(np.float64)
        s["X"] = X
        del s["epochs"]
    return sessions, good_channels, bad


def shift_epoch_with_zeros(epoch: np.ndarray, shift: int) -> np.ndarray:
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


def vote_decode_epoch(dec: SSVEPDecoder, epoch: np.ndarray) -> int:
    votes = {}
    for shift in TEST_VOTE_OFFSETS:
        x_shift = shift_epoch_with_zeros(epoch, int(shift))[np.newaxis, :, :]
        pred = int(dec.decode(x_shift)[0])
        votes[pred] = votes.get(pred, 0) + 1
    return sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def apply_window_batch(X: np.ndarray, start_s: float, end_s: float, srate: int) -> np.ndarray:
    """Apply time window on batch X=(n_trials,n_ch,n_samples)."""
    start = int(round(start_s * srate))
    end = int(round(end_s * srate))
    start = max(0, start)
    end = min(X.shape[2], end)
    if end - start < MIN_WINDOW_SAMPLES:
        return np.empty((X.shape[0], X.shape[1], 0), dtype=X.dtype)
    return X[:, :, start:end]


def train_decoder(X_train: np.ndarray, y_train: np.ndarray, n_components: int, rng: np.random.Generator):
    X_aug, y_aug = augment_training_data(X_train, y_train, rng)
    pt = SSVEPPretrainer(
        srate=ETRCA_SRATE,
        wp=ETRCA_WP,
        ws=ETRCA_WS,
        filter_order=ETRCA_FILTER_ORDER,
        rp=ETRCA_RP,
        n_components=n_components,
        ensemble=ETRCA_ENSEMBLE,
    )
    pt.fit(X_aug, y_aug)
    return SSVEPDecoder(pt)


def evaluate_train_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_components: int,
    rng_seed: int,
) -> Tuple[float, Dict[int, Dict[str, int]]]:
    rng = np.random.default_rng(rng_seed)
    dec = train_decoder(X_train, y_train, n_components=n_components, rng=rng)
    correct = 0
    per_label: Dict[int, Dict[str, int]] = {}
    for i in range(len(y_test)):
        yt = int(y_test[i])
        pred = vote_decode_epoch(dec, X_test[i])
        if yt not in per_label:
            per_label[yt] = {"correct": 0, "total": 0}
        per_label[yt]["total"] += 1
        if pred == yt:
            correct += 1
            per_label[yt]["correct"] += 1
    return correct / len(y_test), per_label


def within_session_loo(X: np.ndarray, y: np.ndarray, n_components: int, rng_seed: int) -> float:
    correct = 0
    n = len(y)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        Xi, yi = X[mask], y[mask]
        Xo, yo = X[i : i + 1], y[i : i + 1]
        acc, _ = evaluate_train_test(Xi, yi, Xo, yo, n_components=n_components, rng_seed=rng_seed + i)
        if acc > 0.5:
            correct += 1
    return correct / n


def search_best_window_for_session(
    X: np.ndarray, y: np.ndarray, n_components: int, rng_seed: int
) -> Tuple[Tuple[float, float], float]:
    best_window = None
    best_acc = -1.0
    for start_s, end_s in WINDOW_CANDIDATES_S:
        Xw = apply_window_batch(X, start_s, end_s, ETRCA_SRATE)
        if Xw.shape[2] == 0:
            continue
        acc = within_session_loo(Xw, y, n_components=n_components, rng_seed=rng_seed)
        if acc > best_acc:
            best_acc = acc
            best_window = (start_s, end_s)
    if best_window is None:
        raise RuntimeError("没有可用窗口，请检查 WINDOW_CANDIDATES_S。")
    return best_window, best_acc


def print_matrix(session_names: List[str], matrix: np.ndarray, title: str):
    print(title)
    header = "train\\test".ljust(30) + "".join([n[:14].ljust(16) for n in session_names])
    print(header)
    print("-" * len(header))
    for i, name in enumerate(session_names):
        row = name[:28].ljust(30)
        for j in range(len(session_names)):
            row += f"{matrix[i, j]:.4f}".ljust(16)
        print(row)


def main():
    print("=" * 90)
    print("eTRCA Session 报告 + 跨 Session 验证矩阵")
    print("=" * 90)
    print(f"数据集匹配: {PRETRAIN_DATASET_GLOB}")
    print(f"预处理: BP {BANDPASS_LOW}-{BANDPASS_HIGH}Hz, notch={NOTCH_FREQS}, offset={T_OFFSET_S}s, fs={FS_TARGET}")
    print(f"训练增强: {TRAIN_AUG_ENABLED}, copies={TRAIN_AUG_COPIES}, jitter=±{TRAIN_JITTER_MAX_SAMPLES}")
    print(f"测试投票偏移: {TEST_VOTE_OFFSETS}")
    print(f"候选窗口(裁剪后): {WINDOW_CANDIDATES_S}")
    print()

    sessions, good_channels, _bad = load_sessions()
    session_names = [s["name"] for s in sessions]

    print("Session 数据概览:")
    for s in sessions:
        print(f"  {s['name']}: X={s['X'].shape}, labels={sorted(np.unique(s['y']).tolist())}")
    print(f"统一通道: {good_channels}")
    print()

    for n_comp in ETRCA_N_COMPONENTS_GRID:
        print("=" * 90)
        print(f"n_components={n_comp}")
        print("=" * 90)

        print("1) Session 内 leave-one-trial-out:")
        within_scores = []
        best_windows = []
        for idx, s in enumerate(sessions):
            t0 = time.time()
            best_w, acc = search_best_window_for_session(
                s["X"], s["y"], n_components=n_comp, rng_seed=RNG_SEED + idx * 1000
            )
            within_scores.append(acc)
            best_windows.append(best_w)
            print(
                f"  {s['name']}: best_window=[{best_w[0]:.2f},{best_w[1]:.2f}) "
                f"acc={acc:.4f}  (耗时 {time.time() - t0:.1f}s)"
            )
        print(f"  平均: {float(np.mean(within_scores)):.4f}")
        print()

        print("2) 跨 Session 验证矩阵 (train i 的最佳窗 -> test j):")
        n_s = len(sessions)
        mat = np.zeros((n_s, n_s), dtype=np.float64)
        for i in range(n_s):
            w_i = best_windows[i]
            Xi = apply_window_batch(sessions[i]["X"], w_i[0], w_i[1], ETRCA_SRATE)
            yi = sessions[i]["y"]
            for j in range(n_s):
                Xj = apply_window_batch(sessions[j]["X"], w_i[0], w_i[1], ETRCA_SRATE)
                yj = sessions[j]["y"]
                if i == j:
                    mat[i, j] = within_scores[i]
                else:
                    acc, _ = evaluate_train_test(
                        Xi, yi, Xj, yj, n_components=n_comp, rng_seed=RNG_SEED + i * 100 + j
                    )
                    mat[i, j] = acc
        print_matrix(session_names, mat, title="准确率矩阵:")
        print()


if __name__ == "__main__":
    main()
