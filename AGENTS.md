# Repository Guidelines
## Self-update rule
After major code changes, update this AGENTS.md to reflect:
- architecture
- conventions
- dependencies
## Project Structure & Module Organization
This repository is a ROS2 workspace for Unity-integrated BCI workflows.
- `src/eeg_processing/`: core EEG/BCI logic (P300 + SSVEP controllers, decoding pipeline, model assets).
- `src/eeg_processing/eeg_processing/utils.py`: shared SSVEP utility module for thread-safe EEG ring buffers, enum-based node states, and per-trial state dataclasses.
- `src/eeg_processing/eeg_processing/ssvep_communication_node2_config.py`: static config module for `SSVEP_Communication_Node2.py`; edit defaults here instead of expanding ROS parameter declarations.
- `src/eeg_processing/eeg_processing/SSVEP_Communication_Node2.py`: refactored SSVEP communication node that keeps decode/pretrain behavior, reads most defaults from the config module, and only exposes a small runtime override surface through ROS parameters.
- `src/eeg_processing/eeg_processing/SSVEP_Communication_Node3.py`: modular SSVEP communication node; keep only node wiring (parameters, pub/sub, timer dispatch, lifecycle cleanup) in this file.
- `src/eeg_processing/eeg_processing/SSVEP_Communication_Node3_1.py`: Node3 v1 variant wiring entry; uses `_1` companion modules and adds decode debug decoupling toggle.
- `src/eeg_processing/eeg_processing/SSVEP_Communication_Node4_test.py`: Node4_test wiring entry for runtime FBCCA decoding + reasoner flow; keep mode wiring here and put behavior in `_2_test` modules.
- `src/eeg_processing/eeg_processing/pretrain_2_test.py`: Node4_test pretrain/shared EEG module; pretrain stage is data-recording only (no model training), keeps EEG/trigger capture and dataset persistence.
- `src/eeg_processing/eeg_processing/decode_2_test.py`: Node4_test decode module; uses runtime FBCCA via a unified preprocess/decode interface and keeps UI 8-slot mapping alignment for reasoner selection.
- `src/eeg_processing/eeg_processing/ssvep_runtime_fbcca.py`: Node4_test runtime algorithm component; centralizes 1_2-style preprocessing (demean/detrend/highpass/notch + resample) and FBCCA decode APIs (`preprocess_epoch`, `decode_epoch`) for controller reuse.
- `src/eeg_processing/eeg_processing/metabci_stim.py`: MetaBCI demo-derived stimulus entry; keeps official multi-paradigm layout and adds Node4-style EEG+label recording for the standard SSVEP paradigm.
- `src/eeg_processing/eeg_processing/decode.py`: decode-mode module for trial preparation, image publication, decode command publishing, and decode state transitions.
- `src/eeg_processing/eeg_processing/decode_1.py`: Node3_1 decode module; publishes explicit decode batch envelope commands (`batch_start`/`batch_end`) around `count<=6` image packets.
- `src/eeg_processing/eeg_processing/pretrain.py`: pretrain-mode module plus shared EEG TCP/trigger/epoch capture and dataset persistence logic used by decode/pretrain.
- `src/eeg_processing/eeg_processing/pretrain_1.py`: Node3_1 shared EEG module; supports full EEG/trigger bypass debug mode while keeping decode/pretrain state flow runnable.
- `src/eeg_processing/eeg_processing/reasoner.py`: reasoner handshake, grouped image intake, history publishing, and selection/rollback command flow.
- `src/eeg_processing/eeg_processing/reasoner_1.py`: Node3_1 reasoner module copy paired with `_1` decode/pretrain modules; keeps staged action stack for slot-7 undo (`confirm` -> `rollback`, `selection` -> `undo_selection`).
- `src/eeg_processing/eeg_processing/ssvep_communication_node3_config.py`: static defaults for Node3 (general/unity/trigger/eeg/decode/pretrain/reasoner); keep runtime override surface minimal.
- `src/eeg_processing/eeg_processing/LLMStreamManager.cs`: Unity-side LLM stream text panel subscriber; consumes JSON string events from `/llm_output_stream` and updates TMP UI on the Unity main thread.
- `src/publisher_test/`: utility/test publishers, UDP trigger sender, TCP listener.
- `src/publisher_test/publisher_test/reasoner_publish_test_1.py`: v1 reasoner test publisher with fixed 4-group sizes (6/5/4/3) and explicit `batch_start`/`batch_end` envelope commands.
- `src/publisher_test/publisher_test/reasoner_publish_test_2.py`: multi-stage (A/B/C) reasoner publisher that scans `picture/`, fuses camera1/camera2 object crops with camera-priority dedup, sends paged batches (`<=6`) for object/category/activity, renders text candidates into images, and supports OpenAI-compatible activity generation with local fallback.
- `src/ROS-TCP-Endpoint/`: Unity ROS TCP bridge package (`ros_tcp_endpoint`).
- `data/`: recorded trials, mappings, and generated datasets/plots.
- `dev_logs/`: development notes.

Keep node code inside each package module (for example `src/eeg_processing/eeg_processing/*.py`) and package metadata in `package.xml`, `setup.py`, and `setup.cfg`.

## Build, Test, and Development Commands
Run from workspace root:
- `colcon build --symlink-install`: build all packages with editable install.
- `colcon build --packages-select eeg_processing`: build one package.
- `source install/setup.bash`: load built packages into shell.
- `colcon test --packages-select eeg_processing publisher_test ros_tcp_endpoint`: run package tests.
- `colcon test-result --verbose`: inspect failures.
- `ros2 launch ros_tcp_endpoint endpoint.py`: start Unity TCP endpoint.
- `ros2 run eeg_processing central_controller_ssvep_node2 --ros-args -p run_mode:=decode`: example runtime command.

## Coding Style & Naming Conventions
- Python: follow PEP 8, 4-space indentation, and PEP 257 docstrings.
- Lint gates are defined via `ament_flake8` and `ament_pep257`; keep code clean enough to pass both.
- ROS node/module naming in this repo typically uses descriptive snake_case files with CamelCase class names (for example `CentralControllerSSVEPNode2.py`).
- For new SSVEP controller work, prefer shared helpers from `eeg_processing/utils.py` over redefining buffer/state helpers inside each node file.
- New controller state machines should use `NodeState` enums and trial reset helpers instead of ad hoc string literals and repeated field reinitialization.
- For `SSVEP_Communication_Node2.py`, static defaults belong in `ssvep_communication_node2_config.py`; only high-frequency runtime toggles such as mode/reasoner/debug overrides should remain as ROS parameters.
- For `SSVEP_Communication_Node3.py`, keep the same small ROS parameter surface (`run_mode`, `reasoner_mode_enabled`, `mock_selected_index`, `save_dir`, `image_dir`, `decode_max_trials`) and put all other defaults in `ssvep_communication_node3_config.py`.
- For `SSVEP_Communication_Node3_1.py`, the runtime surface extends with `eeg_bypass_debug`; when enabled, `_1` modules bypass EEG TCP ingest and trigger UDP send for reasoner/Unity-only debugging.
- For Node3 maintenance, prefer editing behavior in `decode.py` / `pretrain.py` / `reasoner.py` and avoid moving mode logic back into the main node file.
- For Node3_1 maintenance, edit behavior in `decode_1.py` / `pretrain_1.py` / `reasoner_1.py`; keep only wiring in `SSVEP_Communication_Node3_1.py`.
- For Node4_test maintenance, edit behavior in `decode_2_test.py` / `pretrain_2_test.py` / `reasoner_2_test.py`; keep only wiring/parameters in `SSVEP_Communication_Node4_test.py`.
- For `metabci_stim.py`, keep SSVEP flashing/index/rest timing consistent with MetaBCI demo logic; only extend trial lifecycle with trigger(1/2)-aligned EEG epoch recording.
- `metabci_stim.py` SSVEP recording outputs should follow Node4-style naming (`ssvep4_pretrain_trials_*`, `ssvep4_pretrain_metadata_*`, `ssvep4_pretrain_dataset_*`) to stay analysis-compatible.
- Node4_test decode slot conventions use full UI slots `0..7`: image slots `0,1,2,4,5,6`, confirm `3`, undo/rollback `7`; decode label mapping must stay aligned with reasoner slot semantics.
- Node4_test is fixed to runtime FBCCA (no eTRCA model training/loading path); keep algorithm internals inside `ssvep_runtime_fbcca.py`, and keep controller files calling only unified runtime APIs.
- Node4_test preprocessing convention follows `1_2_preprocess.py` chain at runtime: demean -> detrend -> highpass 2Hz -> notch 50/100Hz at acquisition rate, then resample to decode target srate.
- Node4_test decode/preprocess defaults are config-level (`FBCCARuntimeConfig` in `ssvep_communication_node4_test_config.py`); keep filterbank and preprocessing parameters in config rather than hardcoding in node logic.
- Node4_test FBCCA runtime supports per-epoch dynamic bad-channel handling (MAD-based scale outlier check) via `FBCCARuntimeConfig` (`bad_channel_*` fields); default strategy is `bad_channel_mode=drop` to exclude detected bad channels from decode input (with `bad_channel_min_channels` guard).
- Node4_test pretrain completion should only save dataset/metadata CSV+NPY and must not auto-train or emit model sidecars.
- Decode v1 batch protocol (Node3_1 -> Unity): publish `cmd=batch_start;trial=...;target=...;count=...` on `/ssvep_decode_cmd`, then image packets on `/image_seg`, then `cmd=batch_end;...`; Unity should flash only active dynamic slots implied by `count` (max 6).
- Reasoner v1 test protocol (`reasoner_publish_test_1` -> Node3_1): each group is enclosed by `cmd=batch_start;group=...;count=...` and `cmd=batch_end;group=...;count=...` on `/reasoner/images`; image frames keep `group/index/image_path` metadata and can carry `count`/`end` for backward compatibility.
- Reasoner v2 staged protocol (`reasoner_publish_test_2` -> Node3_1): image frame metadata extends with `stage/page/item_uid/item_label`; `cmd=confirm` means page-forward/stage-forward navigation (if no next page, it is ignored with log), `cmd=rollback` means page-backward navigation based on confirm history (cross-stage allowed) and clears selections only on the rollback target page, `cmd=undo_selection` means cancel a prior staged selection by `stage/page/item_uid` and republish that page, `cmd=reuse_page` means keep current page and restart decode `prepare` without re-publishing images (used by A-stage object selection), and only `cmd=done` closes the session.
- Reasoner v2 LLM stream protocol (`reasoner_publish_test_2*` -> Node3_1 -> Unity): reasoner publishes JSON events on `/reasoner/llm_stream` using `std_msgs/String` (`type=reset|append|done|error`, `stage=activity`, optional `text`); Node3_1 forwards to `/llm_output_stream` for Unity display.
- Node3_1 reasoner history conventions: `reasoner_1.py` appends/publishes history images for staged selections (`object/category/activity`) except immediate duplicate re-selections (same stage + same item_uid as stack top), which are skipped and directly restart decode `prepare`; slot `7` now undoes the latest action from an action stack (`confirm` -> publish `rollback`, `selection` -> remove matching history item + publish `undo_selection` + send history UDP `delete_last`).
- `reasoner_publish_test_2` camera conflict policy is code-level (`PREFERRED_CAMERA`) rather than ROS parameters; default is `camera2` priority.
- `reasoner_publish_test_2` LLM settings are code-level constants (`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_TIMEOUT_S`) and use OpenAI-compatible Chat Completions JSON; fallback activity candidates must remain available when LLM output is missing/invalid.
- Console entry points should remain explicit and task-oriented (see each package `setup.py`).

## Testing Guidelines
- Test framework: `pytest` with ROS ament linters.
- Location: each package `test/` directory.
- Naming: `test_*.py` files and `test_*` functions.
- For communication-node refactors, at minimum validate `python -m py_compile` on the touched module plus `colcon build --packages-select eeg_processing` from the workspace root.
- Before PRs, run both package build and `colcon test`; include `colcon test-result --verbose` output when fixing failures.

## Commit & Pull Request Guidelines
- Current history favors short, focused commit subjects (often concise Chinese/English phrases). Keep one logical change per commit.
- Preferred commit format: imperative summary, optionally scoped (example: `eeg_processing: refine UDP listener timeout`).
- PRs should include: purpose, impacted packages, how to run/verify, and sample logs or screenshots when Unity-facing behavior changes.
- Link related issue/task IDs and note any parameter/port changes (for example UDP `9999`, TCP `10000`).
