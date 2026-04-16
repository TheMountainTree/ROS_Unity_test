#!/usr/bin/env python3
"""分析预训练 dataset 中同一 label 下 trial 间的内部一致性（两两 Pearson 相关）。"""

import numpy as np
from itertools import combinations


def main():
    data_path = "data/central_controller_ssvep_node4_test/ssvep4_pretrain_dataset_20260414_161545.npy"
    d = np.load(data_path, allow_pickle=True).item()
    x = d["x"]  # object array, each (8, ~2020)
    y = d["y"]  # int32 labels 1-8

    labels = sorted(np.unique(y).tolist())
    n_channels = x[0].shape[0]

    # 对齐到最短长度，避免长度不一致导致相关性计算问题
    # 将每个 epoch 降采样到 256Hz 后截断到统一长度
    from scipy.signal import resample
    target_fs = 256
    orig_fs = 1000

    resampled = {}
    for i in range(len(x)):
        label = int(y[i])
        epoch = x[i]  # (8, ~2020)
        n_target = int(epoch.shape[1] * target_fs / orig_fs)
        ep = resample(epoch, n_target, axis=1)
        if label not in resampled:
            resampled[label] = []
        resampled[label].append(ep)

    # 每个 label 截断到该 label 下最短的 trial 长度
    for label in resampled:
        min_len = min(e.shape[1] for e in resampled[label])
        resampled[label] = [e[:, :min_len] for e in resampled[label]]

    print("=" * 70)
    print("SSVEP Pretrain Dataset — 同 Label Trial 内部一致性分析")
    print("=" * 70)
    print(f"数据集: {data_path}")
    print(f"总 trial 数: {len(x)}, labels: {labels}")
    print(f"每个 label 的 trial 数: {[len(resampled[l]) for l in labels]}")
    print(f"通道数: {n_channels}, 降采样率: {target_fs}Hz")
    print()

    overall_scores = []

    for label in labels:
        trials = resampled[label]  # list of (8, min_len)
        n_trials = len(trials)
        print(f"--- Label {label} ({n_trials} trials) ---")

        # 展平为 (8*min_len,) 向量做两两相关
        flat = [t.flatten() for t in trials]

        # 两两 Pearson 相关矩阵
        corr_matrix = np.zeros((n_trials, n_trials))
        for i in range(n_trials):
            for j in range(n_trials):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    r = np.corrcoef(flat[i], flat[j])[0, 1]
                    corr_matrix[i, j] = r

        # 打印矩阵
        header = "        " + "  ".join(f"T{j+1:>3}" for j in range(n_trials))
        print(header)
        for i in range(n_trials):
            row = "  ".join(f"{corr_matrix[i, j]:6.3f}" for j in range(n_trials))
            print(f"  T{i+1:>3}   {row}")

        # 统计指标
        upper_indices = list(combinations(range(n_trials), 2))
        upper_vals = [corr_matrix[i, j] for i, j in upper_indices]
        mean_r = np.mean(upper_vals)
        std_r = np.std(upper_vals)
        min_r = np.min(upper_vals)
        max_r = np.max(upper_vals)
        median_r = np.median(upper_vals)

        print(f"  两两相关统计: mean={mean_r:.4f}, std={std_r:.4f}, "
              f"min={min_r:.4f}, max={max_r:.4f}, median={median_r:.4f}")
        print()

        overall_scores.append({"label": label, "mean": mean_r, "std": std_r,
                               "min": min_r, "max": max_r, "median": median_r,
                               "n_pairs": len(upper_vals)})

    # 总结
    print("=" * 70)
    print("各 Label 一致性汇总")
    print("=" * 70)
    print(f"{'Label':>6}  {'Mean r':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}  {'Median':>8}  {'Pairs':>6}")
    print("-" * 60)
    for s in overall_scores:
        print(f"  {s['label']:>4}   {s['mean']:8.4f}  {s['std']:8.4f}  "
              f"{s['min']:8.4f}  {s['max']:8.4f}  {s['median']:8.4f}  {s['n_pairs']:>6}")

    all_means = [s["mean"] for s in overall_scores]
    print(f"\n全局平均一致性 (各 label mean 的均值): {np.mean(all_means):.4f}")


if __name__ == "__main__":
    main()
