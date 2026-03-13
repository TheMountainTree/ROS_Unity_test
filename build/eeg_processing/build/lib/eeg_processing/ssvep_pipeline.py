#!/usr/bin/env python3
"""
SSVEP Pipeline — Modular Components
====================================
Four portable, reusable components for SSVEP‑eTRCA processing:

1. **SSVEPDataLoader**  — Load MetaBCI SSVEP datasets.
2. **SSVEPPretrainer**  — Train FBTRCA (eTRCA) and persist projection matrices.
3. **SSVEPDecoder**     — Decode new EEG epochs using a pre‑trained model.
4. **SSVEPEvaluator**   — Evaluate decoding accuracy (single‑shot & cross‑validation).

SSVEPPretrainer and SSVEPDecoder are designed to be **directly portable** into
production ROS nodes (e.g. CentrlControllerSSVEPNode2) — they depend only on
numpy, scipy, and metabci.brainda, with no ROS or MNE dependency.
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# MNE environment setup (only affects SSVEPDataLoader; safe to import elsewhere)
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
    pass  # MNE not needed for Pretrainer / Decoder

# MetaBCI imports — core algorithm components (no torch dependency)
try:
    from metabci.brainda.algorithms.decomposition.base import generate_filterbank
    from metabci.brainda.algorithms.decomposition.cca import FBTRCA
except ImportError as exc:
    print(f"Error: Could not import metabci.brainda core algorithms: {exc}")
    sys.exit(1)

# Lazy-imported in SSVEPDataLoader and SSVEPEvaluator (require torch):
#   metabci.brainda.datasets.Nakanishi2015
#   metabci.brainda.paradigms.SSVEP
#   metabci.brainda.algorithms.utils.model_selection.*


# =========================================================================
# 1. SSVEPDataLoader
# =========================================================================
@dataclass
class SSVEPDataLoader:
    """Load SSVEP data from MetaBCI datasets.

    Parameters
    ----------
    srate : int
        Target sampling rate (Hz).
    duration : float
        Epoch duration in seconds.
    t_offset : float
        Epoch start offset in seconds (visual latency compensation).
    channels : list[str]
        EEG channel names to keep.
    dataset_class
        MetaBCI dataset class (default: ``Nakanishi2015``).
    """

    srate: int = 256
    duration: float = 3.0
    t_offset: float = 0.15
    channels: List[str] = field(default_factory=lambda: ["O1", "Oz", "O2", "POz"])
    dataset_class: Any = None  # default filled in __post_init__

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
        """Load and return ``(X, y, meta)`` for the given subject.

        Returns
        -------
        X : ndarray, shape (n_trials, n_channels, n_samples)
        y : ndarray, shape (n_trials,)
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

        # Optional hooks (progress tracking)
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


# =========================================================================
# 2. SSVEPPretrainer
# =========================================================================
@dataclass
class SSVEPPretrainer:
    """Train an FBTRCA (eTRCA) model and persist it.

    This class is **portable** — it has no dependency on ROS, MNE data‑loading,
    or any other infrastructure.  It can be directly imported and used in a
    production ROS node.

    Parameters
    ----------
    srate : int
        Sampling rate used for filterbank design.
    wp : list[tuple]
        Passband edges for each sub‑band, e.g. ``[(6, 50), (14, 50), (22, 50)]``.
    ws : list[tuple]
        Stopband edges for each sub‑band.
    filter_order : int
        Butterworth filter order.
    rp : float
        Maximum ripple in the passband (dB).
    n_components : int
        Number of spatial‑filter components per sub‑band.
    ensemble : bool
        ``True`` to use ensemble TRCA (eTRCA).
    n_jobs : int
        Parallelism for FBTRCA.
    """

    srate: int = 256
    wp: List[tuple] = field(default_factory=lambda: [(6, 50), (14, 50), (22, 50)])
    ws: List[tuple] = field(default_factory=lambda: [(4, 52), (12, 52), (20, 52)])
    filter_order: int = 4
    rp: float = 0.5
    n_components: int = 1
    ensemble: bool = True
    n_jobs: int = 1

    def __post_init__(self):
        self._filterbank = generate_filterbank(
            self.wp, self.ws, srate=self.srate, order=self.filter_order, rp=self.rp
        )
        self._filterweights = np.array(
            [(n + 1) ** (-1.25) + 0.25 for n in range(len(self.wp))]
        )
        self._estimator: Optional[FBTRCA] = None

    # ------------------------------------------------------------------
    @property
    def filterbank(self) -> np.ndarray:
        """The SOS filterbank array."""
        return self._filterbank

    @property
    def filterweights(self) -> np.ndarray:
        """Sub‑band weighting coefficients."""
        return self._filterweights

    @property
    def estimator(self) -> Optional[FBTRCA]:
        """The underlying FBTRCA estimator (``None`` before ``fit``)."""
        return self._estimator

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SSVEPPretrainer":
        """Fit the eTRCA model on training data.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
        y : ndarray, shape (n_trials,)

        Returns
        -------
        self
        """
        self._estimator = FBTRCA(
            filterbank=self._filterbank,
            n_components=self.n_components,
            ensemble=self.ensemble,
            filterweights=self._filterweights,
            n_jobs=self.n_jobs,
        )
        self._estimator.fit(X=X, y=y)
        return self

    def save(self, filepath: str) -> None:
        """Persist the trained model (incl. projection matrices) to *filepath*.

        Parameters
        ----------
        filepath : str
            Destination path (e.g. ``model.pkl``).  Parent dirs are created
            automatically.
        """
        if self._estimator is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "estimator": self._estimator,
            "filterbank": self._filterbank,
            "filterweights": self._filterweights,
            "config": {
                "srate": self.srate,
                "wp": self.wp,
                "ws": self.ws,
                "filter_order": self.filter_order,
                "rp": self.rp,
                "n_components": self.n_components,
                "ensemble": self.ensemble,
            },
        }
        with open(filepath, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath: str) -> "SSVEPPretrainer":
        """Load a previously saved model and return a ready‑to‑use Pretrainer.

        Parameters
        ----------
        filepath : str
            Path to a ``.pkl`` file produced by :meth:`save`.
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
            n_components=cfg["n_components"],
            ensemble=cfg["ensemble"],
        )
        obj._estimator = payload["estimator"]
        obj._filterbank = payload["filterbank"]
        obj._filterweights = payload["filterweights"]
        return obj


# =========================================================================
# 3. SSVEPDecoder
# =========================================================================
class SSVEPDecoder:
    """Decode SSVEP epochs using a pre‑trained eTRCA model.

    This class is **portable** — it can be directly imported and used in a
    production ROS node for real‑time decoding.

    Usage
    -----
    >>> decoder = SSVEPDecoder.from_file("model.pkl")
    >>> labels = decoder.decode(X_test)

    Or from an already‑fitted pretrainer:

    >>> decoder = SSVEPDecoder(pretrainer)
    >>> labels = decoder.decode(X_test)
    """

    def __init__(self, pretrainer: SSVEPPretrainer):
        if pretrainer.estimator is None:
            raise RuntimeError(
                "The pretrainer has not been fitted. Call pretrainer.fit() first."
            )
        self._pretrainer = pretrainer
        self._estimator = pretrainer.estimator

    @classmethod
    def from_file(cls, filepath: str) -> "SSVEPDecoder":
        """Create a decoder by loading a pre‑trained model from disk.

        Parameters
        ----------
        filepath : str
            Path to a ``.pkl`` file produced by :meth:`SSVEPPretrainer.save`.
        """
        pretrainer = SSVEPPretrainer.load(filepath)
        return cls(pretrainer)

    @property
    def pretrainer(self) -> SSVEPPretrainer:
        """The underlying pretrainer holding the fitted model."""
        return self._pretrainer

    def decode(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for new EEG epochs.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)

        Returns
        -------
        labels : ndarray, shape (n_trials,)
        """
        return self._estimator.predict(X)


# =========================================================================
# 4. SSVEPEvaluator
# =========================================================================
class SSVEPEvaluator:
    """Evaluate SSVEP decoding results.

    Provides single‑shot accuracy computation, confusion matrix, and a
    convenience method for k‑fold cross‑validation.
    """

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Compute evaluation metrics.

        Returns
        -------
        dict with keys:
            - ``accuracy`` : float
            - ``n_correct`` : int
            - ``n_total`` : int
            - ``per_class_accuracy`` : dict[label → float]
            - ``confusion_matrix`` : ndarray  (rows=true, cols=pred)
            - ``labels`` : ndarray of sorted unique labels
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
        """Run k‑fold cross‑validation.

        Parameters
        ----------
        X, y, meta
            Data as returned by :meth:`SSVEPDataLoader.load`.
        pretrainer_config
            Keyword arguments forwarded to :class:`SSVEPPretrainer`.
            If ``None``, default parameters are used.
        kfold : int
            Number of folds.
        random_seed : int
            Random seed for reproducibility.

        Returns
        -------
        dict with keys:
            - ``fold_accuracies`` : list[float]
            - ``fold_times`` : list[float]
            - ``mean_accuracy`` : float
            - ``mean_fold_time`` : float
            - ``fold_details`` : list[dict]  (per‑fold evaluation dicts)
        """
        if pretrainer_config is None:
            pretrainer_config = {}

        np.random.seed(random_seed)

        # Try metabci's model_selection (requires torch); fall back to manual k-fold
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
            # Manual stratified k-fold (pure numpy, no torch needed)
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

            pretrainer = SSVEPPretrainer(**pretrainer_config)
            pretrainer.fit(X[train_ind], y[train_ind])

            decoder = SSVEPDecoder(pretrainer)
            y_pred = decoder.decode(X[test_ind])

            result = SSVEPEvaluator.evaluate(y[test_ind], y_pred)
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
    def print_report(results: Dict[str, Any], title: str = "Evaluation Report") -> None:
        """Pretty‑print a cross‑validation or single‑shot result dict."""
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")

        if "fold_accuracies" in results:
            for i, (acc, t) in enumerate(
                zip(results["fold_accuracies"], results["fold_times"])
            ):
                print(f"  Fold {i + 1}: acc={acc:.4f}  time={t:.2f}s")
            print(f"  {'─' * 40}")
            print(f"  Mean Accuracy : {results['mean_accuracy']:.4f}")
            print(f"  Mean Fold Time: {results['mean_fold_time']:.2f}s")
        else:
            print(f"  Accuracy : {results['accuracy']:.4f}")
            print(f"  Correct  : {results['n_correct']}/{results['n_total']}")
            if "per_class_accuracy" in results:
                print(f"  Per‑class:")
                for lbl, acc in results["per_class_accuracy"].items():
                    print(f"    class {lbl}: {acc:.4f}")

        print(f"{'=' * 60}\n")
