#!/usr/bin/env python3
"""
SSVEP FBSCCA 处理模块 — 非预训练方法与 eTRCA 性能对比
========================================================
提供四个可移植、可复用的 SSVEP-FBSCCA 处理组件：

1. **SSVEPDataLoaderFBSCCA**  — 加载 MetaBCI SSVEP 数据集
2. **SSVEPPretrainerFBSCCA**  — 配置 FBSCCA 参考信号（无需训练）
3. **SSVEPDecoderFBSCCA**     — 使用 FBSCCA 解码（基于参考信号，无预训练）
4. **SSVEPEvaluatorFBSCCA**   — 评估解码精度（单次 & 交叉验证）

本模块镜像了 `ssvep_pipeline.py` 中的 eTRCA 实现，但使用
滤波器组标准典型相关分析（FBSCCA）替代滤波器组任务相关成分分析（FBTRCA/eTRCA）。

FBSCCA vs eTRCA：
- FBSCCA：滤波器组标准 CCA 方法，使用预定义的正弦参考信号进行典型相关分析
          完全不需要训练数据，仅依赖目标频率列表即可工作
          适合零训练场景或实时系统初始化阶段
- eTRCA：集成 TRCA 方法，需要训练数据来学习空间滤波器
         在有足够训练数据时通常能达到更高的精度

SSVEPPretrainerFBSCCA 和 SSVEPDecoderFBSCCA 设计为可直接移植到
生产环境的 ROS 节点中 — 它们仅依赖 numpy、scipy 和 metabci.brainda，
不依赖 ROS 或 MNE。
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# MNE 环境配置（仅影响 SSVEPDataLoaderFBSCCA，其他组件可安全导入）
# ---------------------------------------------------------------------------
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MNE_DATA", "/tmp/mne_data")
os.makedirs(os.environ["MNE_DATA"], exist_ok=True)
os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")

try:
    import mne
    import mne.datasets.utils

    _original_do_path_update = mne.datasets.utils._do_path_update

    def _skip_path_update(path, update_path, key, sign):
        return

    mne.datasets.utils._do_path_update = _skip_path_update
except ImportError:
    pass  # Pretrainer 和 Decoder 不需要 MNE

# MetaBCI 导入 — 核心算法组件（无 torch 依赖）
try:
    from metabci.brainda.algorithms.decomposition.base import generate_filterbank
    from metabci.brainda.algorithms.decomposition.cca import FBSCCA
except ImportError as exc:
    print(f"错误：无法导入 metabci.brainda 核心算法：{exc}")
    sys.exit(1)

# 以下模块在 SSVEPDataLoaderFBSCCA 和 SSVEPEvaluatorFBSCCA 中延迟导入（需要 torch）：
#   metabci.brainda.datasets.Nakanishi2015
#   metabci.brainda.paradigms.SSVEP
#   metabci.brainda.algorithms.utils.model_selection.*


# =========================================================================
# 1. SSVEPDataLoaderFBSCCA
# =========================================================================
@dataclass
class SSVEPDataLoaderFBSCCA:
    """从 MetaBCI 数据集加载 SSVEP 数据用于 FBSCCA 处理。

    参数
    ----------
    srate : int
        目标采样率（Hz）
    duration : float
        数据片段时长（秒）
    t_offset : float
        数据片段起始偏移（秒），用于补偿视觉延迟
    channels : list[str]
        需要保留的脑电通道名称列表
    dataset_class
        MetaBCI 数据集类（默认：``Nakanishi2015``）
    """

    srate: int = 256
    duration: float = 3.0
    t_offset: float = 0.15
    channels: List[str] = field(default_factory=lambda: ["O1", "Oz", "O2", "POz"])
    dataset_class: Any = None  # 默认值在 __post_init__ 中填充

    def __post_init__(self):
        if self.dataset_class is None:
            from metabci.brainda.datasets import Nakanishi2015
            self.dataset_class = Nakanishi2015

    def load(
        self,
        subject_id: int = 1,
        n_jobs: int = 1,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Any]:
        """加载并返回指定被试的 ``(X, y, meta)`` 数据。

        返回
        -------
        X : ndarray, 形状为 (n_trials, n_channels, n_samples)
        y : ndarray, 形状为 (n_trials,)
        meta : DataFrame
        """
        from metabci.brainda.paradigms import SSVEP

        intervals = [(self.t_offset, self.t_offset + self.duration)]
        dataset = self.dataset_class()

        paradigm = SSVEP(
            channels=self.channels,
            intervals=intervals,
            srate=self.srate,
        )

        # 可选钩子（进度跟踪）
        def epochs_hook(epochs, caches):
            caches["epoch_stage"] = caches.get("epoch_stage", -1) + 1
            return epochs, caches

        def data_hook(X, y, meta, caches):
            caches["data_stage"] = caches.get("data_stage", -1) + 1
            return X, y, meta, caches

        paradigm.register_epochs_hook(epochs_hook)
        paradigm.register_data_hook(data_hook)

        X, y, meta = paradigm.get_data(
            dataset,
            subjects=[subject_id],
            return_concat=True,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        return X, y, meta

    def get_reference_signals(
        self,
        n_samples: int,
        freqs: Optional[List[float]] = None,
        n_harmonics: int = 5,
    ) -> np.ndarray:
        """生成 FBSCCA 所需的参考信号。

        参数
        ----------
        n_samples : int
            每个数据片段的时间采样点数
        freqs : list[float], 可选
            目标频率列表（Hz），如果为 None 则使用 Nakanishi2015 数据集的频率
        n_harmonics : int
            需要包含的谐波数量

        返回
        -------
        Y_ref : ndarray, 形状为 (n_freqs, 2*n_harmonics, n_samples)
            每个目标频率的参考信号
        """
        if freqs is None:
            # Nakanishi2015 数据集的默认频率（12 个目标频率）
            freqs = [9.25, 11.25, 13.25, 9.75, 11.75, 13.75,
                     10.25, 12.25, 14.25, 10.75, 12.75, 14.75]

        n_freqs = len(freqs)
        t = np.arange(n_samples) / self.srate
        Y_ref = np.zeros((n_freqs, 2 * n_harmonics, n_samples))

        for i, f in enumerate(freqs):
            for h in range(n_harmonics):
                # 正弦分量
                Y_ref[i, 2 * h, :] = np.sin(2 * np.pi * (h + 1) * f * t)
                # 余弦分量
                Y_ref[i, 2 * h + 1, :] = np.cos(2 * np.pi * (h + 1) * f * t)

        return Y_ref


# =========================================================================
# 2. SSVEPPretrainerFBSCCA
# =========================================================================
@dataclass
class SSVEPPretrainerFBSCCA:
    """配置 FBSCCA 模型（非训练方法，仅设置参考信号和滤波器组）。

    FBSCCA 使用滤波器组和参考信号（正弦模板）进行典型相关分析，
    完全不需要训练数据来学习任何参数。这使得它成为真正的"零训练"方法。

    本类是可移植的 — 不依赖 ROS、MNE 数据加载或任何其他基础设施。
    可以直接导入并在生产环境的 ROS 节点中使用。

    参数
    ----------
    srate : int
        用于滤波器组设计和参考信号生成的采样率
    wp : list[tuple]
        每个子带的通带边界，如 ``[(6, 50), (14, 50), (22, 50)]``
    ws : list[tuple]
        每个子带的阻带边界
    filter_order : int
        巴特沃斯滤波器阶数
    rp : float
        通带内最大纹波（dB）
    n_harmonics : int
        参考信号的谐波数量
    n_components : int
        使用的 CCA 成分数量
    freqs : list[float], 可选
        目标频率列表（Hz），如果为 None 则使用 Nakanishi2015 数据集的频率
    n_jobs : int
        FBSCCA 的并行度
    """

    srate: int = 256
    wp: List[tuple] = field(default_factory=lambda: [(6, 50), (14, 50), (22, 50)])
    ws: List[tuple] = field(default_factory=lambda: [(4, 52), (12, 52), (20, 52)])
    filter_order: int = 4
    rp: float = 0.5
    n_harmonics: int = 5
    n_components: int = 1
    freqs: Optional[List[float]] = None
    n_jobs: int = 1

    def __post_init__(self):
        self._filterbank = generate_filterbank(
            self.wp, self.ws, srate=self.srate, order=self.filter_order, rp=self.rp
        )
        self._filterweights = np.array(
            [(n + 1) ** (-1.25) + 0.25 for n in range(len(self.wp))]
        )
        self._estimator: Optional[FBSCCA] = None
        self._n_samples: Optional[int] = None
        self._Y_ref: Optional[np.ndarray] = None

        # Nakanishi2015 数据集的默认频率（12 个目标频率）
        # 频率列表：9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75 Hz
        if self.freqs is None:
            self.freqs = [9.25, 11.25, 13.25, 9.75, 11.75, 13.75,
                          10.25, 12.25, 14.25, 10.75, 12.75, 14.75]

    # ------------------------------------------------------------------
    @property
    def filterbank(self) -> np.ndarray:
        """SOS 滤波器组数组"""
        return self._filterbank

    @property
    def filterweights(self) -> np.ndarray:
        """子带权重系数"""
        return self._filterweights

    @property
    def estimator(self) -> Optional[FBSCCA]:
        """底层的 FBSCCA 估计器（在 ``fit`` 之前为 ``None``）"""
        return self._estimator

    @property
    def Y_ref(self) -> Optional[np.ndarray]:
        """用于 CCA 的参考信号"""
        return self._Y_ref

    # ------------------------------------------------------------------
    def _generate_reference_signals(
        self,
        n_samples: int,
        freqs: List[float],
    ) -> np.ndarray:
        """生成用于 CCA 的正弦参考信号。

        参数
        ----------
        n_samples : int
            时间采样点数
        freqs : list[float]
            目标频率列表（Hz）

        返回
        -------
        Y_ref : ndarray, 形状为 (n_freqs, 2*n_harmonics, n_samples)
        """
        n_freqs = len(freqs)
        t = np.arange(n_samples) / self.srate
        Y_ref = np.zeros((n_freqs, 2 * self.n_harmonics, n_samples))

        for i, f in enumerate(freqs):
            for h in range(self.n_harmonics):
                # 正弦分量
                Y_ref[i, 2 * h, :] = np.sin(2 * np.pi * (h + 1) * f * t)
                # 余弦分量
                Y_ref[i, 2 * h + 1, :] = np.cos(2 * np.pi * (h + 1) * f * t)

        return Y_ref

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SSVEPPretrainerFBSCCA":
        """配置 FBSCCA 模型。

        注意：FBSCCA 是非训练方法，此方法仅用于：
        1. 确定数据采样点数
        2. 确定类别标签并映射到频率
        3. 生成参考信号

        FBSCCA 不会从 X 和 y 中学习任何参数。

        参数
        ----------
        X : ndarray, 形状为 (n_trials, n_channels, n_samples)
            用于确定采样点数和类别数量
        y : ndarray, 形状为 (n_trials,)
            用于确定类别标签

        返回
        -------
        self
        """
        n_trials, n_channels, n_samples = X.shape
        self._n_samples = n_samples

        # 获取唯一类别标签并映射到频率
        unique_labels = np.sort(np.unique(y))
        n_classes = len(unique_labels)

        # 将标签映射到频率索引（假设标签从 0 或 1 开始）
        # 对于 Nakanishi2015：标签通常对应频率索引
        if n_classes <= len(self.freqs):
            # 使用对应标签的频率
            self._class_freqs = [self.freqs[int(lbl) % len(self.freqs)] for lbl in unique_labels]
        else:
            raise ValueError(
                f"类别数量 ({n_classes}) 超过了频率数量 ({len(self.freqs)})"
            )

        # 为所有类别生成参考信号
        self._Y_ref = self._generate_reference_signals(n_samples, self._class_freqs)

        # 创建 FBSCCA 估计器
        self._estimator = FBSCCA(
            filterbank=self._filterbank,
            n_components=self.n_components,
            filterweights=self._filterweights,
            n_jobs=self.n_jobs,
        )

        # FBSCCA 的 fit 方法
        self._estimator.fit(X=X, y=y, Yf=self._Y_ref)
        self._labels = unique_labels

        return self

    def save(self, filepath: str) -> None:
        """将配置好的模型（包括参考信号）持久化保存到 *filepath*。

        参数
        ----------
        filepath : str
            目标路径（如 ``model.pkl``），父目录会自动创建
        """
        if self._estimator is None:
            raise RuntimeError("模型尚未配置，请先调用 fit() 方法")

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "estimator": self._estimator,
            "filterbank": self._filterbank,
            "filterweights": self._filterweights,
            "Y_ref": self._Y_ref,
            "n_samples": self._n_samples,
            "class_freqs": getattr(self, "_class_freqs", None),
            "labels": getattr(self, "_labels", None),
            "config": {
                "srate": self.srate,
                "wp": self.wp,
                "ws": self.ws,
                "filter_order": self.filter_order,
                "rp": self.rp,
                "n_harmonics": self.n_harmonics,
                "n_components": self.n_components,
                "freqs": self.freqs,
            },
        }
        with open(filepath, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath: str) -> "SSVEPPretrainerFBSCCA":
        """加载先前保存的模型并返回可用的 Pretrainer。

        参数
        ----------
        filepath : str
            由 :meth:`save` 生成的 ``.pkl`` 文件路径
        """
        with open(filepath, "rb") as f:
            payload = pickle.load(f)

        cfg = payload["config"]
        obj = cls(
            srate=cfg["srate"],
            wp=cfg["wp"],
            ws=cfg["ws"],
            filter_order=cfg["filter_order"],
            rp=cfg["rp"],
            n_harmonics=cfg["n_harmonics"],
            n_components=cfg["n_components"],
            freqs=cfg["freqs"],
        )
        obj._estimator = payload["estimator"]
        obj._filterbank = payload["filterbank"]
        obj._filterweights = payload["filterweights"]
        obj._Y_ref = payload.get("Y_ref")
        obj._n_samples = payload.get("n_samples")
        obj._class_freqs = payload.get("class_freqs")
        obj._labels = payload.get("labels")
        return obj


# =========================================================================
# 3. SSVEPDecoderFBSCCA
# =========================================================================
class SSVEPDecoderFBSCCA:
    """使用 FBSCCA 解码 SSVEP 数据片段（非预训练方法）。

    本类使用滤波器组标准 CCA 方法，基于预定义的参考信号进行解码，
    完全不需要训练数据。是可移植的 — 可以直接导入并在生产环境的
    ROS 节点中进行实时解码。

    使用方法
    --------
    >>> decoder = SSVEPDecoderFBSCCA.from_file("model.pkl")
    >>> labels = decoder.decode(X_test)

    或从已配置的 pretrainer 创建：

    >>> decoder = SSVEPDecoderFBSCCA(pretrainer)
    >>> labels = decoder.decode(X_test)
    """

    def __init__(self, pretrainer: SSVEPPretrainerFBSCCA):
        if pretrainer.estimator is None:
            raise RuntimeError(
                "pretrainer 尚未配置，请先调用 pretrainer.fit() 方法"
            )
        self._pretrainer = pretrainer
        self._estimator = pretrainer.estimator

    @classmethod
    def from_file(cls, filepath: str) -> "SSVEPDecoderFBSCCA":
        """从磁盘加载配置好的模型创建解码器。

        参数
        ----------
        filepath : str
            由 :meth:`SSVEPPretrainerFBSCCA.save` 生成的 ``.pkl`` 文件路径
        """
        pretrainer = SSVEPPretrainerFBSCCA.load(filepath)
        return cls(pretrainer)

    @property
    def pretrainer(self) -> SSVEPPretrainerFBSCCA:
        """持有已配置模型的底层 pretrainer"""
        return self._pretrainer

    def decode(self, X: np.ndarray) -> np.ndarray:
        """预测 EEG 数据片段的类别标签。

        参数
        ----------
        X : ndarray, 形状为 (n_trials, n_channels, n_samples)

        返回
        -------
        labels : ndarray, 形状为 (n_trials,)
        """
        return self._estimator.predict(X)


# =========================================================================
# 4. SSVEPEvaluatorFBSCCA
# =========================================================================
class SSVEPEvaluatorFBSCCA:
    """评估 FBSCCA 的 SSVEP 解码结果。

    提供单次精度计算、混淆矩阵以及 k 折交叉验证的便捷方法。
    """

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """计算评估指标。

        返回
        -------
        包含以下键的字典：
            - ``accuracy`` : float，准确率
            - ``n_correct`` : int，正确数量
            - ``n_total`` : int，总数量
            - ``per_class_accuracy`` : dict[label → float]，每类准确率
            - ``confusion_matrix`` : ndarray，混淆矩阵（行=真实，列=预测）
            - ``labels`` : ndarray，排序后的唯一标签
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        n_total = len(y_true)
        n_correct = int(np.sum(y_true == y_pred))
        accuracy = n_correct / n_total if n_total > 0 else 0.0

        labels = np.sort(np.unique(np.concatenate([y_true, y_pred])))
        label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for yt, yp in zip(y_true, y_pred):
            cm[label_to_idx[yt], label_to_idx[yp]] += 1

        per_class = {}
        for lbl in labels:
            mask = y_true == lbl
            if mask.sum() > 0:
                per_class[lbl] = float(np.mean(y_pred[mask] == lbl))
            else:
                per_class[lbl] = 0.0

        return {
            "accuracy": accuracy,
            "n_correct": n_correct,
            "n_total": n_total,
            "per_class_accuracy": per_class,
            "confusion_matrix": cm,
            "labels": labels,
        }

    @staticmethod
    def cross_validate(
        X: np.ndarray,
        y: np.ndarray,
        meta: Any,
        pretrainer_config: Optional[Dict[str, Any]] = None,
        kfold: int = 6,
        random_seed: int = 38,
    ) -> Dict[str, Any]:
        """运行 k 折交叉验证。

        参数
        ----------
        X, y, meta
            由 :meth:`SSVEPDataLoaderFBSCCA.load` 返回的数据
        pretrainer_config
            传递给 :class:`SSVEPPretrainerFBSCCA` 的关键字参数
            如果为 ``None``，则使用默认参数
        kfold : int
            折数
        random_seed : int
            随机种子，用于可重复性

        返回
        -------
        包含以下键的字典：
            - ``fold_accuracies`` : list[float]，每折准确率
            - ``fold_times`` : list[float]，每折耗时
            - ``mean_accuracy`` : float，平均准确率
            - ``mean_fold_time`` : float，平均每折耗时
            - ``fold_details`` : list[dict]，每折的详细评估结果
        """
        if pretrainer_config is None:
            pretrainer_config = {}

        np.random.seed(random_seed)

        # 尝试使用 metabci 的 model_selection（需要 torch）；回退到手动 k 折
        try:
            from metabci.brainda.algorithms.utils.model_selection import (
                set_random_seeds,
                generate_kfold_indices,
                match_kfold_indices,
            )
            set_random_seeds(random_seed)
            _indices = generate_kfold_indices(meta, kfold=kfold)

            def _get_fold(k):
                train_ind, val_ind, test_ind = match_kfold_indices(k, meta, _indices)
                return np.concatenate((train_ind, val_ind)), test_ind

        except (ImportError, ModuleNotFoundError):
            # 手动分层 k 折（纯 numpy，无需 torch）
            n = len(y)
            all_indices = np.arange(n)
            np.random.shuffle(all_indices)
            fold_size = n // kfold
            _fold_splits = []
            for k in range(kfold):
                start = k * fold_size
                end = start + fold_size if k < kfold - 1 else n
                _fold_splits.append(all_indices[start:end])

            def _get_fold(k):
                test_ind = _fold_splits[k]
                train_ind = np.concatenate([_fold_splits[j] for j in range(kfold) if j != k])
                return train_ind, test_ind

        fold_accs: List[float] = []
        fold_times: List[float] = []
        fold_details: List[Dict[str, Any]] = []

        for k in range(kfold):
            t0 = time.perf_counter()
            train_ind, test_ind = _get_fold(k)

            # 注意：FBSCCA 不需要训练数据，但为了接口一致性，仍传入训练集来确定类别
            pretrainer = SSVEPPretrainerFBSCCA(**pretrainer_config)
            pretrainer.fit(X[train_ind], y[train_ind])

            decoder = SSVEPDecoderFBSCCA(pretrainer)
            y_pred = decoder.decode(X[test_ind])

            result = SSVEPEvaluatorFBSCCA.evaluate(y[test_ind], y_pred)
            fold_accs.append(result["accuracy"])
            fold_times.append(time.perf_counter() - t0)
            fold_details.append(result)

        return {
            "fold_accuracies": fold_accs,
            "fold_times": fold_times,
            "mean_accuracy": float(np.mean(fold_accs)),
            "mean_fold_time": float(np.mean(fold_times)),
            "fold_details": fold_details,
        }

    @staticmethod
    def print_report(results: Dict[str, Any], title: str = "评估报告") -> None:
        """格式化打印交叉验证或单次评估结果"""
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")

        if "fold_accuracies" in results:
            for i, (acc, t) in enumerate(
                zip(results["fold_accuracies"], results["fold_times"])
            ):
                print(f"  第 {i + 1} 折: 准确率={acc:.4f}  耗时={t:.2f}s")
            print(f"  {'─' * 40}")
            print(f"  平均准确率 : {results['mean_accuracy']:.4f}")
            print(f"  平均耗时   : {results['mean_fold_time']:.2f}s")
        else:
            print(f"  准确率 : {results['accuracy']:.4f}")
            print(f"  正确数: {results['n_correct']}/{results['n_total']}")
            if "per_class_accuracy" in results:
                print(f"  每类准确率:")
                for lbl, acc in results["per_class_accuracy"].items():
                    print(f"    类别 {lbl}: {acc:.4f}")

        print(f"{'=' * 60}\n")


# =========================================================================
# 5. 性能对比工具
# =========================================================================
def compare_etrca_fbscca(
    subject_id: int = 1,
    kfold: int = 6,
    random_seed: int = 38,
    verbose: bool = True,
) -> Dict[str, Any]:
    """在同一数据上对比 eTRCA（预训练方法）和 FBSCCA（非预训练方法）的性能。

    本函数在 Nakanishi2015 数据集上运行两种算法并返回对比性能指标。

    参数
    ----------
    subject_id : int
        要加载的被试 ID
    kfold : int
        交叉验证折数
    random_seed : int
        随机种子，用于可重复性
    verbose : bool
        是否打印详细进度

    返回
    -------
    包含对比结果的字典：
        - ``etrca_cv_results`` : eTRCA 交叉验证结果
        - ``fbscca_cv_results`` : FBSCCA 交叉验证结果
        - ``accuracy_diff`` : eTRCA 准确率 - FBSCCA 准确率
        - ``time_diff`` : eTRCA 耗时 - FBSCCA 耗时
        - ``subject_id`` : 被试 ID
    """
    if verbose:
        print("=" * 70)
        print("  SSVEP 性能对比: eTRCA（预训练） vs FBSCCA（非预训练）")
        print("=" * 70)

    # 加载数据（只加载一次）
    if verbose:
        print("\n[1/3] 加载 Nakanishi2015 数据...")

    loader_fbscca = SSVEPDataLoaderFBSCCA(
        srate=256,
        duration=3.0,
        t_offset=0.15,
        channels=["O1", "Oz", "O2", "POz"],
    )
    X, y, meta = loader_fbscca.load(subject_id=subject_id)

    if verbose:
        print(f"  数据形状 : {X.shape}")
        print(f"  标签     : {np.unique(y)}")
        print(f"  试次数   : {len(y)}")

    # 通用的滤波器组配置
    filterbank_config = {
        "srate": 256,
        "wp": [(6, 50), (14, 50), (22, 50)],
        "ws": [(4, 52), (12, 52), (20, 52)],
    }

    # eTRCA 配置（预训练方法）
    etrca_config = {
        **filterbank_config,
        "n_components": 1,
        "ensemble": True,
    }

    # FBSCCA 配置（非预训练方法）
    fbscca_config = {
        **filterbank_config,
        "n_harmonics": 5,
        "n_components": 1,
    }

    # eTRCA 交叉验证
    if verbose:
        print(f"\n[2/3] 运行 eTRCA（预训练）{kfold} 折交叉验证...")

    try:
        from ssvep_pipeline import (
            SSVEPPretrainer,
            SSVEPDecoder,
            SSVEPEvaluator,
        )
    except ImportError:
        from eeg_processing.ssvep_pipeline import (
            SSVEPPretrainer,
            SSVEPDecoder,
            SSVEPEvaluator,
        )

    etrca_cv_results = SSVEPEvaluator.cross_validate(
        X, y, meta,
        pretrainer_config=etrca_config,
        kfold=kfold,
        random_seed=random_seed,
    )

    if verbose:
        SSVEPEvaluator.print_report(
            etrca_cv_results, title=f"eTRCA（预训练）— 被试 {subject_id}"
        )

    # FBSCCA 交叉验证
    if verbose:
        print(f"[3/3] 运行 FBSCCA（非预训练）{kfold} 折交叉验证...")

    fbscca_cv_results = SSVEPEvaluatorFBSCCA.cross_validate(
        X, y, meta,
        pretrainer_config=fbscca_config,
        kfold=kfold,
        random_seed=random_seed,
    )

    if verbose:
        SSVEPEvaluatorFBSCCA.print_report(
            fbscca_cv_results, title=f"FBSCCA（非预训练）— 被试 {subject_id}"
        )

    # 汇总对比
    accuracy_diff = etrca_cv_results["mean_accuracy"] - fbscca_cv_results["mean_accuracy"]
    time_diff = etrca_cv_results["mean_fold_time"] - fbscca_cv_results["mean_fold_time"]

    if verbose:
        print("\n" + "=" * 70)
        print("  对比汇总")
        print("=" * 70)
        print(f"  {'指标':<20} {'eTRCA(预训练)':>14} {'FBSCCA(非预训练)':>14} {'差值':>12}")
        print("  " + "─" * 60)
        print(f"  {'平均准确率':<18} {etrca_cv_results['mean_accuracy']:>14.4f} "
              f"{fbscca_cv_results['mean_accuracy']:>14.4f} {accuracy_diff:>+12.4f}")
        print(f"  {'平均耗时 (s)':<16} {etrca_cv_results['mean_fold_time']:>14.2f} "
              f"{fbscca_cv_results['mean_fold_time']:>14.2f} {time_diff:>+12.2f}")
        print("=" * 70)

        if accuracy_diff > 0.01:
            print(f"  → eTRCA（预训练）比 FBSCCA（非预训练）高 {accuracy_diff:.2%}")
        elif accuracy_diff < -0.01:
            print(f"  → FBSCCA（非预训练）比 eTRCA（预训练）高 {-accuracy_diff:.2%}")
        else:
            print(f"  → 性能相当（差距 < 1%）")
        print("=" * 70 + "\n")

    return {
        "etrca_cv_results": etrca_cv_results,
        "fbscca_cv_results": fbscca_cv_results,
        "accuracy_diff": accuracy_diff,
        "time_diff": time_diff,
        "subject_id": subject_id,
    }


# =========================================================================
# 主测试函数
# =========================================================================
def main():
    """运行 FBSCCA 流水线测试并与 eTRCA 对比"""
    print("=" * 70)
    print("  SSVEP FBSCCA 流水线 — 集成测试（非预训练方法）")
    print("=" * 70)

    subject_id = 1

    # ------------------------------------------------------------------
    # 1. 数据加载
    # ------------------------------------------------------------------
    print("\n[1/6] 使用 SSVEPDataLoaderFBSCCA 加载数据...")
    loader = SSVEPDataLoaderFBSCCA(
        srate=256,
        duration=3.0,
        t_offset=0.15,
        channels=["O1", "Oz", "O2", "POz"],
    )
    X, y, meta = loader.load(subject_id=subject_id)
    print(f"  数据形状 : {X.shape}")
    print(f"  标签     : {np.unique(y)}")
    print(f"  试次数   : {len(y)}")

    # ------------------------------------------------------------------
    # 2. 配置 FBSCCA（非训练方法，仅设置参考信号和滤波器组）
    # ------------------------------------------------------------------
    print("\n[2/6] 配置 FBSCCA 模型（无需训练，仅设置参考信号和滤波器组）...")
    pretrainer = SSVEPPretrainerFBSCCA(
        srate=256,
        wp=[(6, 50), (14, 50), (22, 50)],
        ws=[(4, 52), (12, 52), (20, 52)],
        n_harmonics=5,
        n_components=1,
    )
    pretrainer.fit(X, y)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(script_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    model_path = os.path.join(assets_dir, "ssvep_fbscca_model.pkl")
    pretrainer.save(model_path)
    print(f"  模型配置已保存到: {model_path}")

    # ------------------------------------------------------------------
    # 3. 解码（使用参考信号，无需训练）
    # ------------------------------------------------------------------
    print("\n[3/6] 使用 SSVEPDecoderFBSCCA 解码（基于参考信号，无需训练）...")
    decoder = SSVEPDecoderFBSCCA.from_file(model_path)
    y_pred = decoder.decode(X)

    # 检查解码结果
    train_acc = float(np.mean(y_pred == y))
    print(f"  训练集解码准确率: {train_acc:.4f}")

    # ------------------------------------------------------------------
    # 4. 单次评估
    # ------------------------------------------------------------------
    print("\n[4/6] 使用 SSVEPEvaluatorFBSCCA 评估解码结果...")
    result = SSVEPEvaluatorFBSCCA.evaluate(y, y_pred)
    SSVEPEvaluatorFBSCCA.print_report(result, title="训练集评估 (FBSCCA)")

    # ------------------------------------------------------------------
    # 5. 交叉验证
    # ------------------------------------------------------------------
    print("[5/6] 运行 6 折交叉验证...")
    cv_results = SSVEPEvaluatorFBSCCA.cross_validate(
        X, y, meta,
        pretrainer_config={
            "srate": 256,
            "wp": [(6, 50), (14, 50), (22, 50)],
            "ws": [(4, 52), (12, 52), (20, 52)],
            "n_harmonics": 5,
            "n_components": 1,
        },
        kfold=6,
        random_seed=38,
    )
    SSVEPEvaluatorFBSCCA.print_report(
        cv_results, title=f"被试 {subject_id} — 6 折交叉验证 (FBSCCA)"
    )

    # ------------------------------------------------------------------
    # 6. 验证模型持久化一致性
    # ------------------------------------------------------------------
    print("[6/6] 验证模型持久化...")
    decoder_direct = SSVEPDecoderFBSCCA(pretrainer)
    decoder_loaded = SSVEPDecoderFBSCCA.from_file(model_path)

    y_direct = decoder_direct.decode(X[:10])
    y_loaded = decoder_loaded.decode(X[:10])

    if np.array_equal(y_direct, y_loaded):
        print("  ✓ 保存/加载的模型产生相同的预测结果")
    else:
        print("  ✗ 警告：直接模型和加载模型的预测结果不一致！")

    # ------------------------------------------------------------------
    # 7. 与 eTRCA 对比
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  运行 eTRCA（预训练）vs FBSCCA（非预训练）对比...")
    print("=" * 70)

    comparison = compare_etrca_fbscca(subject_id=subject_id, kfold=6, random_seed=38)

    print("\n所有测试已完成")


if __name__ == "__main__":
    main()