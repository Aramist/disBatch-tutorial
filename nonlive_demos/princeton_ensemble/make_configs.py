import json
from pathlib import Path

import numpy as np
import pyjson5

NUM_MODELS = 10
working_dir = Path("/mnt/home/atanelus/ceph/experiments/princeton_ensemble")
base_config_dir = working_dir / "base_configs"
if not base_config_dir.exists():
    raise FileNotFoundError(f"Base config directory {base_config_dir} not found")

base_config_paths = list(
    filter(lambda p: "ensemble" not in p.stem, base_config_dir.glob("*.json5"))
)
base_ensemble_config_path = base_config_dir / "ensemble_config.json5"
ensemble_config_path = working_dir / "ensemble_config.json5"

config_dir = working_dir / "child_configs"
config_dir.mkdir(exist_ok=True)
model_dir = working_dir / "child_models"
model_dir.mkdir(exist_ok=True)
log_dir = working_dir / "logs"
log_dir.mkdir(exist_ok=True)

training_script_path = working_dir / "train_single_model.sh"
runner_script_path = working_dir / "train_models"
dataset_path = Path(
    "/mnt/home/atanelus/ceph/neurips_datasets/audio/mouseearbud-24m-e3_audio.h5"
)


def make_modified_configs():
    modified_config_paths = []
    for bc in base_config_paths:
        with open(bc, "rb") as f:
            config = pyjson5.load(f)
        for i in range(NUM_MODELS):
            config["GENERAL"]["TORCH_SEED"] = i
            config["GENERAL"]["NUMPY_SEED"] = i
            new_config_path = config_dir / f"{bc.stem}_{i}.json5"
            modified_config_paths.append(new_config_path)
            with open(new_config_path, "w") as f:
                json.dump(config, f, indent=4)
    return modified_config_paths


training_script = f"""#!/bin/bash

config_path=$1
model_dir=$2

source ~/.bashrc
source ~/venvs/general/bin/activate

dataset_path={dataset_path}

hostname; date;
echo "Starting training"

# Run training
python -u -m vocalocator \
    --config $config_path \
    --data $dataset_path \
    --save-path $model_dir
"""


def write_training_script():
    with open(training_script_path, "w") as f:
        f.write(training_script)


def make_runner_script(config_paths):
    output_paths = [model_dir / f"model_{i}" for i in range(len(config_paths))]
    with open(runner_script_path, "w") as f:
        for n, (config_path, output_path) in enumerate(zip(config_paths, output_paths)):
            line = (
                f"( chmod +x {training_script_path} && "
                f"{training_script_path} {config_path} {output_path} "
                f") &> {log_dir}/model_{n}.log\n"
            )
            f.write(line)
    num_jobs = min(20, len(config_paths))
    print(
        "module load disBatch && "
        "mkdir disbatch_logs && "
        f"sbatch -n {num_jobs} -p gpu --gpus-per-task=1 --mem=32GB -c 6 -t 1-0 disBatch -p disbatch_logs/ {runner_script_path};"
    )


def write_ensemble_config():
    with open(base_ensemble_config_path, "rb") as f:
        ensemble_config = pyjson5.load(f)

    ensemble_config["ARCHITECTURE"] = "VocalocatorEnsemble"
    constituent_models = []

    model_config_paths = sorted(list(model_dir.glob("*/config.json")))
    if not model_config_paths:
        print("Model configs not yet created. Skipping ensemble config creation")
        return

    for mcp in model_config_paths:
        with open(mcp, "r") as f:
            constituent_models.append(json.load(f))

    ensemble_config["MODEL_PARAMS"]["OUTPUT_TYPE"] = "ENSEMBLE"
    ensemble_config["MODEL_PARAMS"]["CONSTITUENT_MODELS"] = constituent_models

    with open(ensemble_config_path, "w") as f:
        json.dump(ensemble_config, f, indent=4)


if __name__ == "__main__":
    paths = make_modified_configs()
    write_training_script()
    make_runner_script(paths)
    write_ensemble_config()
