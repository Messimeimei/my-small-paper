# Experiment Configuration

Training configs are stored under `configs/training/<task>/`. Evaluation configs are stored under `configs/evaluation/<task>/<method>/`.

The `configs/sweeps/` directory is reserved for batch experiment definitions that enumerate tasks, methods, seeds, and inference modes. Individual runtime configs remain plain YAML and continue to be accepted by the existing CLI.
