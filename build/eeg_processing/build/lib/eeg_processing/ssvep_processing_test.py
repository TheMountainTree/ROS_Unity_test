#!/usr/bin/env python3
"""
SSVEP Processing Test — Integration Demo
=========================================
Demonstrates the full SSVEP pipeline using modular components:

1. **SSVEPDataLoader**  — load MetaBCI Nakanishi2015 data
2. **SSVEPPretrainer**  — train eTRCA and save projection matrices
3. **SSVEPDecoder**     — load saved model and decode
4. **SSVEPEvaluator**   — evaluate accuracy & cross‑validation

This script serves both as a smoke test and as usage documentation.
"""

import os
import tempfile

import numpy as np

try:
    from eeg_processing.ssvep_pipeline import (
        SSVEPDataLoader,
        SSVEPPretrainer,
        SSVEPDecoder,
        SSVEPEvaluator,
    )
except ImportError:
    from ssvep_pipeline import (
        SSVEPDataLoader,
        SSVEPPretrainer,
        SSVEPDecoder,
        SSVEPEvaluator,
    )


def main():
    print("=" * 60)
    print("  SSVEP Modular Pipeline — Integration Test")
    print("=" * 60)

    subject_id = 1

    # ------------------------------------------------------------------
    # 1. Data Loading
    # ------------------------------------------------------------------
    print("\n[1/5] Loading data with SSVEPDataLoader ...")
    loader = SSVEPDataLoader(
        srate=256,
        duration=3.0,
        t_offset=0.15,
        channels=["O1", "Oz", "O2", "POz"],
    )
    X, y, meta = loader.load(subject_id=subject_id)
    print(f"  Data shape : {X.shape}")
    print(f"  Labels     : {np.unique(y)}")
    print(f"  Num trials : {len(y)}")

    # ------------------------------------------------------------------
    # 2. Pretraining (fit & save)
    # ------------------------------------------------------------------
    print("\n[2/5] Training eTRCA model with SSVEPPretrainer ...")
    pretrainer = SSVEPPretrainer(
        srate=256,
        wp=[(6, 50), (14, 50), (22, 50)],
        ws=[(4, 52), (12, 52), (20, 52)],
        n_components=1,
        ensemble=True,
    )
    pretrainer.fit(X, y)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(script_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    model_path = os.path.join(assets_dir, "ssvep_etrca_model.pkl")
    pretrainer.save(model_path)
    print(f"  Model saved to: {model_path}")

    # ------------------------------------------------------------------
    # 3. Decoding (load saved model & predict)
    # ------------------------------------------------------------------
    print("\n[3/5] Decoding with SSVEPDecoder (loaded from file) ...")
    decoder = SSVEPDecoder.from_file(model_path)
    y_pred = decoder.decode(X)

    # Quick sanity check — decoding training data should be near‑perfect
    train_acc = float(np.mean(y_pred == y))
    print(f"  Train‑set decode accuracy: {train_acc:.4f}")

    # ------------------------------------------------------------------
    # 4. Single-shot Evaluation
    # ------------------------------------------------------------------
    print("\n[4/5] Evaluating decode results with SSVEPEvaluator ...")
    result = SSVEPEvaluator.evaluate(y, y_pred)
    SSVEPEvaluator.print_report(result, title="Train‑Set Evaluation")

    # ------------------------------------------------------------------
    # 5. Cross-Validation
    # ------------------------------------------------------------------
    print("[5/5] Running 6‑fold cross‑validation ...")
    cv_results = SSVEPEvaluator.cross_validate(
        X, y, meta,
        pretrainer_config={
            "srate": 256,
            "wp": [(6, 50), (14, 50), (22, 50)],
            "ws": [(4, 52), (12, 52), (20, 52)],
            "n_components": 1,
            "ensemble": True,
        },
        kfold=6,
        random_seed=38,
    )
    SSVEPEvaluator.print_report(
        cv_results, title=f"Subject {subject_id} — 6‑Fold Cross‑Validation"
    )

    # ------------------------------------------------------------------
    # 6. Verify model persistence consistency
    # ------------------------------------------------------------------
    print("Verifying model persistence ...")
    decoder_direct = SSVEPDecoder(pretrainer)
    decoder_loaded = SSVEPDecoder.from_file(model_path)

    y_direct = decoder_direct.decode(X[:10])
    y_loaded = decoder_loaded.decode(X[:10])

    if np.array_equal(y_direct, y_loaded):
        print("  ✓ Saved/loaded model produces identical predictions.")
    else:
        print("  ✗ WARNING: Predictions differ between direct and loaded model!")

    print("\nAll tests completed.")


if __name__ == "__main__":
    main()
