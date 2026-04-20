#!/usr/bin/env python3
"""Runtime FBCCA component for Node4_test.

This module isolates EEG preprocessing and FBCCA decoding behind stable
interfaces so controller logic does not depend on algorithm internals.
"""

from __future__ import annotations

import inspect
import os
from typing import Iterable, List, Optional, Sequence

import numpy as np
from scipy import signal

# Import FBCCARuntimeConfig from the single authoritative config module.
try:
    from .ssvep_communication_node4_test_config import FBCCARuntimeConfig
except ImportError:
    from ssvep_communication_node4_test_config import FBCCARuntimeConfig

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MNE_DATA", "/tmp/mne_data")
os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")
os.makedirs(os.environ["MNE_DATA"], exist_ok=True)

try:
    from metabci.brainda.algorithms.decomposition import (
        FBSCCA,
        generate_filterbank,
        generate_cca_references,
    )
except ImportError:
    FBSCCA = None  # type: ignore[assignment,misc]
    generate_filterbank = None  # type: ignore[assignment]
    generate_cca_references = None  # type: ignore[assignment]


class SSVEPFBCCARuntime:
    """Runtime wrapper exposing preprocess/decode interfaces."""

    def __init__(
        self,
        config: FBCCARuntimeConfig,
        frequencies: Sequence[float],
        logger=None,
    ) -> None:
        if (
            generate_filterbank is None
            or FBSCCA is None
            or generate_cca_references is None
        ):
            raise RuntimeError(
                "metabci.brainda is required for FBCCA runtime decoding but is not available"
            )

        self.cfg = config
        self.logger = logger
        self.frequencies = [float(v) for v in frequencies]
        if not self.frequencies:
            raise ValueError("frequencies must not be empty")

        self._filterbank = generate_filterbank(
            [tuple(p) for p in self.cfg.wp],
            [tuple(s) for s in self.cfg.ws],
            srate=self.cfg.target_srate,
            order=self.cfg.filter_order,
            rp=self.cfg.rp,
        )
        self._filterweights = np.array(
            [(n + 1) ** (-1.25) + 0.25 for n in range(len(self.cfg.wp))]
        )

        self._estimator: Optional[FBSCCA] = None
        self._estimator_n_channels: int = -1
        self._estimator_n_samples: int = -1
        self._manual_drop_indices: List[int] = self._resolve_manual_drop_indices()

    def preprocess_epoch(self, epoch: np.ndarray, input_fs: float) -> np.ndarray:
        """Apply 1_2-style preprocessing, then resample to target_srate.

        Steps:
        1. Demean (per channel)
        2. Detrend (linear)
        3. Band-pass
        4. Notch 50/100Hz
        5. Resample to target_srate
        """
        if epoch.ndim != 2:
            raise ValueError(
                f"epoch must be 2D (n_channels, n_samples), got shape={epoch.shape}"
            )
        if input_fs <= 0.0:
            raise ValueError(f"input_fs must be > 0, got {input_fs}")

        out = epoch.astype(np.float64, copy=True)

        # 1) Demean
        out -= np.mean(out, axis=1, keepdims=True)

        # 2) Detrend
        out = signal.detrend(out, axis=1)

        # 3) Band-pass
        nyquist = 0.5 * float(input_fs)
        bp_low = max(0.1, float(self.cfg.bandpass_low_hz))
        bp_high = min(float(self.cfg.bandpass_high_hz), nyquist - 0.1)
        if bp_low < bp_high:
            sos_bp = signal.butter(
                int(self.cfg.bandpass_order),
                [bp_low, bp_high],
                btype="bandpass",
                fs=float(input_fs),
                output="sos",
            )
            out = signal.sosfiltfilt(sos_bp, out, axis=1)

        # 4) Notch
        for f0 in self.cfg.notch_freqs_hz:
            f0 = float(f0)
            if 0.0 < f0 < nyquist:
                b, a = signal.iirnotch(
                    w0=f0, Q=float(self.cfg.notch_q), fs=float(input_fs)
                )
                out = signal.filtfilt(b, a, out, axis=1)

        # 5) Resample
        target_fs = float(self.cfg.target_srate)
        if abs(target_fs - float(input_fs)) > 1e-6:
            n_samples = max(1, int(round(out.shape[1] * target_fs / float(input_fs))))
            out = signal.resample(out, n_samples, axis=1)

        return out.astype(np.float32, copy=False)

    def decode_epoch(
        self,
        epoch: np.ndarray,
        input_fs: float,
        active_ui_slots: Optional[Iterable[int]] = None,
    ) -> int:
        """Decode one EEG epoch and return predicted label in 1..N.

        Returns -1 when decoding fails or predicts an inactive image slot.
        """
        # 1) Manual bad-channel exclusion first (on raw epoch)
        epoch_cleaned = self._apply_manual_channel_exclusion(epoch)

        # 2) Preprocess after channel exclusion
        preprocessed = self.preprocess_epoch(epoch_cleaned, input_fs)
        n_channels, n_samples = preprocessed.shape
        if n_samples < 16:
            self._log_warn(f"epoch too short after preprocess: samples={n_samples}")
            return -1

        self._ensure_estimator(n_channels, n_samples)

        try:
            X = preprocessed[np.newaxis, :, :]
            pred_raw = int(self._estimator.predict(X)[0])
        except Exception as exc:
            text = str(exc).lower()
            if "not positive definite" in text:
                try:
                    # Numerical fallback for rare ill-conditioned epochs.
                    jitter = 1e-6
                    rng = np.random.default_rng(42)
                    X_jitter = X + rng.normal(loc=0.0, scale=jitter, size=X.shape)
                    pred_raw = int(self._estimator.predict(X_jitter)[0])
                    self._log_warn(f"FBCCA decode recovered with jitter={jitter:.1e}")
                except Exception:
                    self._log_warn(f"FBCCA decode failed: {exc}")
                    return -1
            else:
                self._log_warn(f"FBCCA decode failed: {exc}")
                return -1

        # MetaBCI FBSCCA in this project/runtime returns 0-based class indices.
        # Map index -> label(1..N) first. Keep a guarded fallback for 1-based output.
        if 0 <= pred_raw < len(self.frequencies):
            predicted_label = pred_raw + 1
        elif pred_raw == len(self.frequencies):
            # Fallback path for rare 1-based outputs from other variants.
            predicted_label = pred_raw
            self._log_warn(
                "FBCCA returned 1-based style class value; "
                "runtime expects 0-based index output."
            )
        else:
            self._log_warn(f"invalid FBCCA prediction value: {pred_raw}")
            return -1

        if active_ui_slots is not None:
            ui_slot = predicted_label - 1
            ui_image_slots = {0, 1, 2, 4, 5, 6}
            if ui_slot in ui_image_slots and ui_slot not in set(active_ui_slots):
                self._log_warn(
                    f"predicted inactive image slot={ui_slot}, active={sorted(set(active_ui_slots))}"
                )
                return -1

        return predicted_label

    def _resolve_manual_drop_indices(self) -> List[int]:
        """Map configured channel names to indices for manual exclusion."""
        channel_order = list(getattr(self.cfg, "channel_name_order", []) or [])
        bad_names = [
            str(v).strip() for v in (getattr(self.cfg, "manual_bad_channels", []) or [])
        ]
        bad_names = [v for v in bad_names if v]
        if not bad_names or not channel_order:
            return []

        name_to_idx = {name.upper(): idx for idx, name in enumerate(channel_order)}
        drop_idx: List[int] = []
        unknown: List[str] = []
        for name in bad_names:
            idx = name_to_idx.get(name.upper())
            if idx is None:
                unknown.append(name)
                continue
            drop_idx.append(int(idx))

        if unknown:
            self._log_warn(
                f"manual_bad_channels contains unknown names {unknown}; "
                f"channel_name_order={channel_order}"
            )

        unique_idx = sorted(set(drop_idx))
        if unique_idx:
            self._log_info(
                f"FBCCA manual channel exclusion enabled: names={bad_names}, indices={unique_idx}"
            )
        return unique_idx

    def _apply_manual_channel_exclusion(self, epoch: np.ndarray) -> np.ndarray:
        """Apply manual channel exclusion from configured channel names."""
        if epoch.ndim != 2 or epoch.shape[0] <= 1:
            return epoch
        if not self._manual_drop_indices:
            return epoch

        valid_drop = [
            idx for idx in self._manual_drop_indices if 0 <= idx < epoch.shape[0]
        ]
        if not valid_drop:
            return epoch

        keep_idx = [i for i in range(epoch.shape[0]) if i not in set(valid_drop)]
        if len(keep_idx) < 2:
            self._log_warn(
                f"manual channel exclusion skipped: keep={len(keep_idx)} < 2, "
                f"drop_indices={valid_drop}"
            )
            return epoch

        return epoch[np.asarray(keep_idx, dtype=np.int32), :].astype(
            np.float32, copy=False
        )

    def _ensure_estimator(self, n_channels: int, n_samples: int) -> None:
        if (
            self._estimator is not None
            and self._estimator_n_channels == n_channels
            and self._estimator_n_samples == n_samples
        ):
            return

        y_labels = np.arange(1, len(self.frequencies) + 1, dtype=np.int32)
        X_dummy = np.zeros((len(y_labels), n_channels, n_samples), dtype=np.float64)
        Y_ref = self._build_reference_signals(n_samples)

        est = FBSCCA(
            filterbank=self._filterbank,
            n_components=int(self.cfg.n_components),
            filterweights=self._filterweights,
            n_jobs=int(self.cfg.n_jobs),
        )
        est.fit(X=X_dummy, y=y_labels, Yf=Y_ref)

        self._estimator = est
        self._estimator_n_channels = n_channels
        self._estimator_n_samples = n_samples
        self._log_info(
            "FBCCA estimator prepared: "
            f"channels={n_channels}, samples={n_samples}, target_srate={self.cfg.target_srate}"
        )

    def _build_reference_signals(self, n_samples: int) -> np.ndarray:
        """Build CCA reference signals with backward/forward API compatibility."""
        target_srate = float(self.cfg.target_srate)
        duration_s = n_samples / target_srate
        n_harmonics = int(self.cfg.n_harmonics)

        # Different metabci versions use different parameter names:
        # - old: generate_cca_references(..., T=...)
        # - newer variants may expose duration=...
        try:
            sig = inspect.signature(generate_cca_references)
            if "duration" in sig.parameters:
                y_ref = generate_cca_references(
                    freqs=self.frequencies,
                    srate=target_srate,
                    duration=duration_s,
                    phases=None,
                    n_harmonics=n_harmonics,
                )
            else:
                y_ref = generate_cca_references(
                    freqs=self.frequencies,
                    srate=target_srate,
                    T=duration_s,
                    phases=None,
                    n_harmonics=n_harmonics,
                )
        except Exception as exc:
            self._log_warn(
                f"generate_cca_references failed ({exc}), fallback to manual refs"
            )
            y_ref = self._generate_reference_signals(n_samples)

        # Align exact sample length to decoder epoch length.
        y_ref = np.asarray(y_ref, dtype=np.float64)
        if y_ref.shape[-1] > n_samples:
            y_ref = y_ref[..., :n_samples]
        elif y_ref.shape[-1] < n_samples:
            pad = n_samples - y_ref.shape[-1]
            y_ref = np.pad(y_ref, ((0, 0), (0, 0), (0, pad)), mode="edge")
        return y_ref

    def _generate_reference_signals(self, n_samples: int) -> np.ndarray:
        """Manual sine/cosine references used as compatibility fallback."""
        t = np.arange(n_samples, dtype=np.float64) / float(self.cfg.target_srate)
        n_freqs = len(self.frequencies)
        n_h = int(self.cfg.n_harmonics)
        y_ref = np.zeros((n_freqs, 2 * n_h, n_samples), dtype=np.float64)
        for i, f in enumerate(self.frequencies):
            for h in range(n_h):
                harm = float(h + 1)
                y_ref[i, 2 * h, :] = np.sin(2.0 * np.pi * harm * f * t)
                y_ref[i, 2 * h + 1, :] = np.cos(2.0 * np.pi * harm * f * t)
        return y_ref

    def _log_info(self, text: str) -> None:
        if self.logger is not None:
            self.logger.info(text)

    def _log_warn(self, text: str) -> None:
        if self.logger is not None:
            self.logger.warning(text)
