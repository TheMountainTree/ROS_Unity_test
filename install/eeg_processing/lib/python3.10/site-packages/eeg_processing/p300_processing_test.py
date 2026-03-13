#!/usr/bin/env python3
import os
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Keep MNE cache/config writes inside writable sandbox paths.
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MNE_DATA", "/tmp/mne_data")
os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")

try:
    # In newer brainda versions, the dataset name is updated.
    from brainda.datasets import Cattan2019 as Cattan_P300
    from brainda.paradigms.p300 import P300
except ImportError:
    raise ImportError("Could not import brainda. Please run 'pip install brainda'.")


BASE_EEG_CHANNELS = [
    "FP1",
    "FP2",
    "FC5",
    "FZ",
    "FC6",
    "T7",
    "CZ",
    "T8",
    "P7",
    "P3",
    "PZ",
    "P4",
    "P8",
    "O1",
    "OZ",
    "O2",
]
TRUE_EOG_CHANNELS = ["VEOG", "HEOG", "EOG1", "EOG2"]


def bandpass_filter_trials(x_trials, fs=100.0, low=0.5, high=30.0, order=4):
    sos = butter(order, [low, high], fs=fs, btype="bandpass", output="sos")
    return [sosfiltfilt(sos, xi, axis=-1) for xi in x_trials]


def baseline_correction_trials(x_trials, baseline_samples=20):
    x_corrected = []
    for xi in x_trials:
        baseline = np.mean(xi[..., :baseline_samples], axis=-1, keepdims=True)
        x_corrected.append(xi - baseline)
    return x_corrected


def remove_artifacts_by_amplitude_threshold(
    x_trials, y_trials, ptp_threshold=120.0, rms_z_threshold=4.0
):
    x_clean_trials = []
    y_clean_trials = []
    for xi, yi in zip(x_trials, y_trials):
        ptp_per_channel = np.ptp(xi, axis=-1)
        max_ptp_per_epoch = np.max(ptp_per_channel, axis=1)

        rms_per_epoch = np.sqrt(np.mean(xi**2, axis=(1, 2)))
        rms_std = np.std(rms_per_epoch)
        if rms_std < 1e-12:
            rms_z = np.zeros_like(rms_per_epoch)
        else:
            rms_z = (rms_per_epoch - np.mean(rms_per_epoch)) / rms_std

        keep_mask = (max_ptp_per_epoch < ptp_threshold) & (
            np.abs(rms_z) < rms_z_threshold
        )
        x_clean_trials.append(xi[keep_mask])
        y_clean_trials.append(yi[keep_mask])
    return x_clean_trials, y_clean_trials


def remove_artifacts_by_eog_regression(x_trials, channel_names):
    eog_indices = [i for i, ch in enumerate(channel_names) if ch in TRUE_EOG_CHANNELS]
    if not eog_indices:
        return None, None

    eeg_indices = [i for i, ch in enumerate(channel_names) if ch not in TRUE_EOG_CHANNELS]
    cleaned_trials = []
    for xi in x_trials:
        xi_clean = xi.copy()
        for epoch_idx in range(xi.shape[0]):
            eog = xi[epoch_idx, eog_indices, :].T
            design = np.column_stack([np.ones(eog.shape[0]), eog])
            for ch_idx in eeg_indices:
                eeg_signal = xi[epoch_idx, ch_idx, :]
                beta, *_ = np.linalg.lstsq(design, eeg_signal, rcond=None)
                artifact = eog @ beta[1:]
                xi_clean[epoch_idx, ch_idx, :] = eeg_signal - artifact
        cleaned_trials.append(xi_clean[:, eeg_indices, :])
    cleaned_channels = [channel_names[i] for i in eeg_indices]
    return cleaned_trials, cleaned_channels


def normalize_trial_labels(yi):
    yi = np.asarray(yi)
    if yi.ndim == 1:
        return yi.astype(int)
    if yi.ndim == 2 and yi.shape[1] == 1:
        return yi[:, 0].astype(int)
    if yi.ndim == 2 and yi.shape[1] > 1:
        return np.argmax(yi, axis=1).astype(int)
    return yi.reshape(-1).astype(int)


def trial_to_features(xi, downsample=2):
    return xi[:, :, ::downsample].reshape(xi.shape[0], -1)


def build_train_set(x_trials, y_trials, trial_indices, downsample=2):
    features = []
    labels = []
    for idx in trial_indices:
        xi = x_trials[idx]
        yi = normalize_trial_labels(y_trials[idx])
        features.append(trial_to_features(xi, downsample=downsample))
        labels.append(yi)
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def get_target_classes_from_code(code):
    code = np.asarray(code).astype(int).reshape(-1)
    return [i + 1 for i, v in enumerate(code) if v == 2]


def stimulus_to_binary_labels(stimulus_ids, code):
    code = np.asarray(code).astype(int).reshape(-1)
    stimulus_ids = normalize_trial_labels(stimulus_ids)
    return np.array([1 if code[int(stim_id) - 1] == 2 else 0 for stim_id in stimulus_ids])


def build_binary_train_set(x_trials, y_trials, meta, trial_indices, downsample=2):
    features = []
    labels = []
    for idx in trial_indices:
        xi = x_trials[idx]
        yi_stim = normalize_trial_labels(y_trials[idx])
        yi_bin = stimulus_to_binary_labels(yi_stim, meta.iloc[idx]["code"])
        features.append(trial_to_features(xi, downsample=downsample))
        labels.append(yi_bin)
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def decode_trial_with_probability_accumulation(classifier, classes, xi, yi, downsample=2):
    epoch_features = trial_to_features(xi, downsample=downsample)
    stimulus_ids = normalize_trial_labels(yi)
    score = {int(class_id): 0.0 for class_id in range(1, 13)}
    pos_idx = int(np.where(classes == 1)[0][0])

    for feat, stim_id in zip(epoch_features, stimulus_ids):
        p = classifier.predict_proba(feat[None, :])[0]
        p_target = float(p[pos_idx])

        score[int(stim_id)] += p_target

    pred_targets = sorted(score.keys(), key=lambda k: score[k], reverse=True)[:2]
    return pred_targets, score


def load_data_with_eog_fallback(subject_id=1):
    dataset = Cattan_P300()
    channels_with_eog = BASE_EEG_CHANNELS + TRUE_EOG_CHANNELS

    try:
        paradigm = P300(channels=channels_with_eog)
        x, y, meta = paradigm.get_data(dataset, subjects=[subject_id], verbose=False)
        channel_names = channels_with_eog
    except Exception:
        paradigm = P300(channels=BASE_EEG_CHANNELS)
        x, y, meta = paradigm.get_data(dataset, subjects=[subject_id], verbose=False)
        channel_names = BASE_EEG_CHANNELS

    return x, y, meta, channel_names


def run_p300_decoding(
    subject_id=1,
    fs=100.0,
    low=0.5,
    high=30.0,
    order=4,
    ptp_threshold=120.0,
    rms_z_threshold=4.0,
    baseline_samples=20,
    downsample=2,
    test_size=0.3,
    random_state=42,
):
    np.random.seed(random_state)
    x, y, meta, channel_names = load_data_with_eog_fallback(subject_id=subject_id)

    x_bandpass = bandpass_filter_trials(x, fs=fs, low=low, high=high, order=order)
    x_artifact_free, cleaned_channels = remove_artifacts_by_eog_regression(
        x_bandpass, channel_names
    )
    if x_artifact_free is None:
        x_artifact_free, y_clean = remove_artifacts_by_amplitude_threshold(
            x_bandpass, y, ptp_threshold=ptp_threshold, rms_z_threshold=rms_z_threshold
        )
        cleaned_channels = channel_names
        removal_method = "Amplitude threshold"
    else:
        y_clean = y
        removal_method = "EOG regression"

    x_baseline = baseline_correction_trials(
        x_artifact_free, baseline_samples=baseline_samples
    )
    trial_ids = np.arange(len(x_baseline))
    train_ids, test_ids = train_test_split(
        trial_ids, test_size=test_size, random_state=random_state, shuffle=True
    )

    X_train, y_train = build_binary_train_set(
        x_baseline, y_clean, meta, train_ids, downsample=downsample
    )
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, multi_class="multinomial"),
    )
    clf.fit(X_train, y_train)
    classes = clf.named_steps["logisticregression"].classes_

    X_test, y_test = build_binary_train_set(
        x_baseline, y_clean, meta, test_ids, downsample=downsample
    )
    y_pred = clf.predict(X_test)
    epoch_acc = float(np.mean(y_pred == y_test))

    trial_results = []
    trial_correct = 0
    for test_trial in test_ids:
        pred_targets, score = decode_trial_with_probability_accumulation(
            clf, classes, x_baseline[test_trial], y_clean[test_trial], downsample=downsample
        )
        true_targets = get_target_classes_from_code(meta.iloc[test_trial]["code"])
        is_correct = set(pred_targets) == set(true_targets)
        if is_correct:
            trial_correct += 1
        trial_results.append(
            {
                "trial_id": int(test_trial),
                "target_class": true_targets,
                "predicted_target_class": pred_targets,
                "trial_score": score,
                "trial_hit": bool(is_correct),
            }
        )

    trial_acc = float(trial_correct / len(test_ids))
    return {
        "subject_id": subject_id,
        "artifact_removal_method": removal_method,
        "channels_used_for_decoding": cleaned_channels,
        "epoch_accuracy_target_nontarget": epoch_acc,
        "trial_accuracy_row_column": trial_acc,
        "trial_results": trial_results,
    }


def main():
    result = run_p300_decoding(subject_id=1)
    for item in result["trial_results"]:
        print("=" * 60)
        print(f"target class: {item['target_class']}")
        print(f"predicted target class: {item['predicted_target_class']}")
        print(f"trial score: {item['trial_score']}")
        print(f"trial hit: {item['trial_hit']}")
    print("=" * 60)
    print(f"total accuracy: {result['trial_accuracy_row_column']:.4f}")


if __name__ == "__main__":
    main()
