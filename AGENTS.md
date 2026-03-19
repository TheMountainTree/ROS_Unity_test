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
- `src/publisher_test/`: utility/test publishers, UDP trigger sender, TCP listener.
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
