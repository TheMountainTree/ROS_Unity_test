#!/usr/bin/env python3
"""
SSVEP FBCCA Pipeline — Modular Components (with Sliding Window)
==============================================================
Three portable, reusable components for SSVEP‑FBSCCA processing:

1. **SSVEPPretrainerFBCCA**  — Configure FBSCCA model (zero‑training, reference
   signal based).  Supports sliding‑window search: slides a fixed‑duration
   window across the epoch to find the optimal position (highest leave‑one‑out
   accuracy), then configures the final model on that window only.

2. **SSVEPDecoderFBCCA**     — Decode new EEG epochs using a configured model.
   Automatically applies the same optimal window learned during pretrain.

3. **SSVEPEvaluatorFBCCA**   — Evaluate decoding accuracy (single‑shot & cross‑validation).

FBSCCA (Filter Bank Standard CCA) vs eTRCA (Ensemble TRCA):
- FBSCCA: Uses predefined sinusoidal reference signals for CCA, zero‑training.
          Suitable for initialization phase or when no training data is available.
- eTRCA: Learns spatial filters from training data, typically higher accuracy
          with sufficient training data.

SSVEPPretrainerFBCCA and SSVEPDecoderFBCCA are designed to be **directly portable**
into production ROS nodes — they depend only on numpy, scipy, and metabci.brainda,
with no ROS or MNE dependency.
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# MetaBCI imports — core algorithm components (no torch dependency)
try:
    from metabci.brainda.algorithms.decomposition.base import generate_filterbank
    from metabci.brainda.algorithms.decomposition.cca import FBSCCA
except ImportError as exc:
    print(f"Error: Could not import metabci.brainda core algorithms: {exc}")
    sys.exit(1)


# =========================================================================
# Helper: Reference Signal Generation
# =========================================================================
def generate_reference_signals(
    n_samples: int,
    freqs: List[float],
    srate: int,
    n_harmonics: int = 5,
) -> np.ndarray:
    """Generate sine/cosine reference signals for CCA.

    Parameters
    ----------
    n_samples : int
        Number of time samples per epoch.
    freqs : list[float]
        Target frequencies (Hz).
    srate : int
        Sampling rate (Hz).
    n_harmonics : int
        Number of harmonics to include.

    Returns
    -------
    Y_ref : ndarray, shape (n_freqs, 2*n_harmonics, n_samples)
        Reference signals for each target frequency.
    """
    n_freqs = len(freqs)
    t = np.arange(n_samples) / srate
    Y_ref = np.zeros((n_freqs, 2 * n_harmonics, n_samples))

    for i, f in enumerate(freqs):
        for h in range(n_harmonics):
            # Sine component
            Y_ref[i, 2 * h, :] = np.sin(2 * np.pi * (h + 1) * f * t)
            # Cosine component
            Y_ref[i, 2 * h + 1, :] = np.cos(2 * np.pi * (h + 1) * f * t)

    return Y_ref


# =========================================================================
# 1. SSVEPPretrainerFBCCA
# =========================================================================
@dataclass
class SSVEPPretrainerFBCCA:
    """Configure an FBSCCA model and persist reference signals.

    Supports sliding‑window search: :meth:`fit_with_window_search` slides a
    fixed‑duration window across the time axis of each epoch, evaluates each
    position via leave‑one‑out accuracy, and picks the best one.  The final
    model is then configured on data from that optimal window only.

    Parameters
    ----------
    srate : int
        Sampling rate used for filterbank design and reference signal generation.
    wp : list[tuple]
        Passband edges for each sub‑band.
    ws : list[tuple]
        Stopband edges for each sub‑band.
    filter_order : int
        Butterworth filter order.
    rp : float
        Maximum ripple in the passband (dB).
    n_harmonics : int
        Number of harmonics for reference signal generation.
    n_components : int
        Number of CCA components per sub‑band.
    freqs : list[float] or None
        Target frequencies (Hz). If None, must be set before fit().
    n_jobs : int
        Parallelism for FBSCCA.
    window_duration_s : float
        Sliding window length in seconds (0 = no windowing, use full epoch).
    window_step_s : float
        Sliding window step in seconds.
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
    window_duration_s: float = 0.0  # 0 = full epoch (no windowing)
    window_step_s: float = 0.125

    def __post_init__(self):
        self._filterbank = generate_filterbank(
            self.wp, self.ws, srate=self.srate, order=self.filter_order, rp=self.rp
        )
        self._filterweights = np.array(
            [(n + 1) ** (-1.25) + 0.25 for n in range(len(self.wp))]
        )
        self._estimator: Optional[FBSCCA] = None
        self._Y_ref: Optional[np.ndarray] = None
        # Optimal window info (set by fit_with_window_search)
        self._optimal_window_start: int = 0
        self._optimal_window_samples: int = 0
        self._window_search_results: Optional[List[Dict[str, Any]]] = None

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
    def estimator(self) -> Optional[FBSCCA]:
        """The underlying FBSCCA estimator (``None`` before ``fit``)."""
        return self._estimator

    @property
    def Y_ref(self) -> Optional[np.ndarray]:
        """Reference signals used for CCA."""
        return self._Y_ref

    @property
    def optimal_window_start(self) -> int:
        """Sample index where the optimal window begins (0‑based)."""
        return self._optimal_window_start

    @property
    def optimal_window_samples(self) -> int:
        """Number of samples in the optimal window."""
        return self._optimal_window_samples

    @property
    def window_search_results(self) -> Optional[List[Dict[str, Any]]]:
        """Detailed results from the sliding‑window search (if run)."""
        return self._window_search_results

    # ------------------------------------------------------------------
    def _make_estimator(self) -> FBSCCA:
        return FBSCCA(
            filterbank=self._filterbank,
            n_components=self.n_components,
            filterweights=self._filterweights,
            n_jobs=self.n_jobs,
        )

    def _get_class_freqs(self, y: np.ndarray) -> List[float]:
        """Map labels to frequencies (labels are 1‑based)."""
        if self.freqs is None:
            raise ValueError(
                "freqs must be set before calling fit(). "
                "Pass freqs=[...] when constructing SSVEPPretrainerFBCCA."
            )
        unique_labels = np.sort(np.unique(y))
        return [self.freqs[int(lbl) - 1] for lbl in unique_labels]

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SSVEPPretrainerFBCCA":
        """Fit the FBSCCA model (generate reference signals and configure estimator).

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
        y : ndarray, shape (n_trials,)

        Returns
        -------
        self
        """
        n_trials, n_channels, n_samples = X.shape
        class_freqs = self._get_class_freqs(y)

        # Generate reference signals
        self._Y_ref = generate_reference_signals(
            n_samples, class_freqs, self.srate, self.n_harmonics
        )

        # Create and fit FBSCCA estimator
        self._estimator = self._make_estimator()
        self._estimator.fit(X=X, y=y, Yf=self._Y_ref)

        # If no window search was done, the window covers the full epoch
        self._optimal_window_samples = n_samples
        self._optimal_window_start = 0
        return self

    # ------------------------------------------------------------------
    def _loo_accuracy(self, X_win: np.ndarray, y: np.ndarray) -> float:
        """Compute leave‑one‑out accuracy for windowed data."""
        n_samples = X_win.shape[2]
        n = len(y)
        correct = 0
        for i in range(n):
            X_train = np.delete(X_win, i, axis=0)
            y_train = np.delete(y, i)
            train_labels = np.sort(np.unique(y_train))
            class_freqs = [self.freqs[int(lbl) - 1] for lbl in train_labels]
            X_test = X_win[i:i + 1]
            y_test = y[i]
            try:
                Y_ref = generate_reference_signals(
                    n_samples, class_freqs, self.srate, self.n_harmonics
                )
                est = self._make_estimator()
                est.fit(X=X_train, y=y_train, Yf=Y_ref)
                pred_idx = int(est.predict(X_test)[0])
                if 0 <= pred_idx < len(train_labels):
                    pred = int(train_labels[pred_idx])
                else:
                    pred = -1
            except Exception:
                pred = -1
            if pred == y_test:
                correct += 1
        return correct / n

    # ------------------------------------------------------------------
    def fit_with_window_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True,
    ) -> "SSVEPPretrainerFBCCA":
        """Slide a fixed‑duration window across epochs to find the optimal
        position, then configure the final FBSCCA model on that window.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
        y : ndarray, shape (n_trials,)
        verbose : bool
            Print progress during search.

        Returns
        -------
        self
        """
        win_samples = int(round(self.window_duration_s * self.srate))
        step_samples = max(1, int(round(self.window_step_s * self.srate)))
        total_samples = X.shape[2]

        if win_samples <= 0 or win_samples >= total_samples:
            if verbose:
                print(f"[WindowSearch] window_duration_s={self.window_duration_s}s "
                      f"covers full epoch or is zero; falling back to full-epoch fit.")
            self._window_search_results = None
            return self.fit(X, y)

        if verbose:
            print(f"[WindowSearch] window={self.window_duration_s}s "
                  f"({win_samples} samples), step={self.window_step_s}s "
                  f"({step_samples} samples), epoch={total_samples} samples "
                  f"({total_samples / self.srate:.2f}s)")

        # Enumerate all valid window start positions
        starts = list(range(0, total_samples - win_samples + 1, step_samples))
        # Ensure the last possible window is included
        if starts[-1] + win_samples < total_samples:
            starts.append(total_samples - win_samples)

        if verbose:
            print(f"[WindowSearch] testing {len(starts)} window positions ...")

        search_results: List[Dict[str, Any]] = []
        best_acc = -1.0
        best_start = 0

        for idx, start in enumerate(starts):
            X_win = X[:, :, start:start + win_samples]
            acc = self._loo_accuracy(X_win, y)

            pos_s = start / self.srate
            search_results.append({
                "window_start_sample": start,
                "window_start_s": pos_s,
                "window_end_s": pos_s + self.window_duration_s,
                "accuracy": acc,
            })

            if acc > best_acc:
                best_acc = acc
                best_start = start

            if verbose:
                bar = "█" * int(acc * 30)
                print(f"  [{idx + 1}/{len(starts)}] "
                      f"start={pos_s:.3f}s  acc={acc:.4f} {bar}")

        self._window_search_results = search_results
        self._optimal_window_start = best_start
        self._optimal_window_samples = win_samples

        best_pos_s = best_start / self.srate
        if verbose:
            print(f"[WindowSearch] best window: start={best_pos_s:.3f}s "
                  f"(sample {best_start}), acc={best_acc:.4f}")

        # Fit final model on the optimal window
        X_optimal = X[:, :, best_start:best_start + win_samples]
        class_freqs = self._get_class_freqs(y)
        self._Y_ref = generate_reference_signals(
            win_samples, class_freqs, self.srate, self.n_harmonics
        )
        self._estimator = self._make_estimator()
        self._estimator.fit(X=X_optimal, y=y, Yf=self._Y_ref)

        if verbose:
            print(f"[WindowSearch] final model configured on "
                  f"samples [{best_start}, {best_start + win_samples})")

        return self

    # ------------------------------------------------------------------
    def extract_window(self, X: np.ndarray) -> np.ndarray:
        """Extract the optimal window from data.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples) or (n_channels, n_samples)

        Returns
        -------
        X_win : ndarray with time axis trimmed to optimal window
        """
        if self._optimal_window_samples <= 0:
            return X
        start = self._optimal_window_start
        end = start + self._optimal_window_samples
        if X.ndim == 3:
            return X[:, :, start:end]
        elif X.ndim == 2:
            return X[:, start:end]
        return X

    # ------------------------------------------------------------------
    def save(self, filepath: str) -> None:
        """Persist the configured model (incl. reference signals and window info)."""
        if self._estimator is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "estimator": self._estimator,
            "filterbank": self._filterbank,
            "filterweights": self._filterweights,
            "Y_ref": self._Y_ref,
            "optimal_window_start": self._optimal_window_start,
            "optimal_window_samples": self._optimal_window_samples,
            "window_search_results": self._window_search_results,
            "config": {
                "srate": self.srate,
                "wp": self.wp,
                "ws": self.ws,
                "filter_order": self.filter_order,
                "rp": self.rp,
                "n_harmonics": self.n_harmonics,
                "n_components": self.n_components,
                "freqs": self.freqs,
                "window_duration_s": self.window_duration_s,
                "window_step_s": self.window_step_s,
            },
        }
        with open(filepath, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath: str) -> "SSVEPPretrainerFBCCA":
        """Load a previously saved model and return a ready‑to‑use Pretrainer."""
        with open(filepath, "rb") as f:
            payload = pickle.load(f)

        cfg = payload["config"]
        obj = cls(
            srate=cfg["srate"],
            wp=[tuple(p) for p in cfg["wp"]],
            ws=[tuple(s) for s in cfg["ws"]],
            filter_order=cfg["filter_order"],
            rp=cfg["rp"],
            n_harmonics=cfg["n_harmonics"],
            n_components=cfg["n_components"],
            freqs=cfg["freqs"],
            window_duration_s=cfg.get("window_duration_s", 0.0),
            window_step_s=cfg.get("window_step_s", 0.125),
        )
        obj._estimator = payload["estimator"]
        obj._filterbank = payload["filterbank"]
        obj._filterweights = payload["filterweights"]
        obj._Y_ref = payload.get("Y_ref")
        obj._optimal_window_start = payload.get("optimal_window_start", 0)
        obj._optimal_window_samples = payload.get("optimal_window_samples", 0)
        obj._window_search_results = payload.get("window_search_results", None)
        return obj


# =========================================================================
# 2. SSVEPDecoderFBCCA
# =========================================================================
class SSVEPDecoderFBCCA:
    """Decode SSVEP epochs using a configured FBSCCA model.

    If the pretrainer was configured with sliding‑window search, the decoder
    automatically extracts the same optimal window before decoding.

    Usage
    -----
    >>> decoder = SSVEPDecoderFBCCA.from_file("model.pkl")
    >>> labels = decoder.decode(X_test)

    Or from an already‑fitted pretrainer:

    >>> decoder = SSVEPDecoderFBCCA(pretrainer)
    >>> labels = decoder.decode(X_test)
    """

    def __init__(self, pretrainer: SSVEPPretrainerFBCCA):
        if pretrainer.estimator is None:
            raise RuntimeError(
                "The pretrainer has not been fitted. Call pretrainer.fit() first."
            )
        self._pretrainer = pretrainer
        self._estimator = pretrainer.estimator

    @classmethod
    def from_file(cls, filepath: str) -> "SSVEPDecoderFBCCA":
        """Create a decoder by loading a configured model from disk."""
        pretrainer = SSVEPPretrainerFBCCA.load(filepath)
        return cls(pretrainer)

    @property
    def pretrainer(self) -> SSVEPPretrainerFBCCA:
        """The underlying pretrainer holding the configured model."""
        return self._pretrainer

    def decode(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for new EEG epochs.

        If the model was configured with sliding‑window search, the optimal
        window is automatically extracted from *X* before prediction.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)

        Returns
        -------
        labels : ndarray, shape (n_trials,)
        """
        X_win = self._pretrainer.extract_window(X)
        return self._estimator.predict(X_win)


# =========================================================================
# 3. SSVEPEvaluatorFBCCA
# =========================================================================
class SSVEPEvaluatorFBCCA:
    """Evaluate SSVEP‑FBSCCA decoding results.

    Provides single‑shot accuracy computation, confusion matrix, and a
    convenience method for k‑fold cross‑validation.
    """

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
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
        meta: Any = None,
        pretrainer_config: Optional[Dict[str, Any]] = None,
        kfold: int = 6,
        random_seed: int = 38,
        use_window_search: bool = False,
    ) -> Dict[str, Any]:
        """Run k‑fold cross‑validation.

        Parameters
        ----------
        X, y, meta
            Data arrays. meta is optional for FBSCCA (used only for metabci k-fold).
        pretrainer_config
            Keyword arguments forwarded to :class:`SSVEPPretrainerFBCCA`.
        kfold : int
            Number of folds.
        random_seed : int
            Random seed for reproducibility.
        use_window_search : bool
            If True, use :meth:`SSVEPPretrainerFBCCA.fit_with_window_search`
            instead of :meth:`fit` in each fold.

        Returns
        -------
        dict with keys:
            - ``fold_accuracies`` : list[float]
            - ``fold_times`` : list[float]
            - ``mean_accuracy`` : float
            - ``mean_fold_time`` : float
            - ``fold_details`` : list[dict]
        """
        if pretrainer_config is None:
            pretrainer_config = {}

        np.random.seed(random_seed)

        # Try metabci's model_selection; fall back to manual k-fold
        use_metabci_kfold = False
        if meta is not None:
            try:
                from metabci.brainda.algorithms.utils.model_selection import (
                    set_random_seeds,
                    generate_kfold_indices,
                    match_kfold_indices,
                )
                set_random_seeds(random_seed)
                _indices = generate_kfold_indices(meta, kfold=kfold)
                use_metabci_kfold = True

                def _get_fold(k):
                    train_ind, val_ind, test_ind = match_kfold_indices(k, meta, _indices)
                    return np.concatenate((train_ind, val_ind)), test_ind

            except (ImportError, ModuleNotFoundError):
                pass

        if not use_metabci_kfold:
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

            pretrainer = SSVEPPretrainerFBCCA(**pretrainer_config)
            if use_window_search and pretrainer.window_duration_s > 0:
                pretrainer.fit_with_window_search(
                    X[train_ind], y[train_ind], verbose=False
                )
            else:
                pretrainer.fit(X[train_ind], y[train_ind])

            decoder = SSVEPDecoderFBCCA(pretrainer)
            y_pred = decoder.decode(X[test_ind])

            result = SSVEPEvaluatorFBCCA.evaluate(y[test_ind], y_pred)
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
