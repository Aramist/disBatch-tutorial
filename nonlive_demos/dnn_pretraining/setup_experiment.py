from pathlib import Path
from typing import Optional

base_dir = Path("/mnt/home/atanelus/ceph/experiments/workshop_dnn_pretraining")
model_dir = base_dir / "models"
base_config_dir = base_dir / "base_configs"
log_dir = base_dir / "logs"

model_dir.mkdir(exist_ok=True, parents=True)
base_config_dir.mkdir(exist_ok=True, parents=True)
log_dir.mkdir(exist_ok=True, parents=True)

pretrain_dataset_dir = Path("/mnt/home/atanelus/ceph/february/pretraining_datasets")
pretrain_dataset_paths = sorted(list(pretrain_dataset_dir.glob("*.h5")))

finetune_dataset_path = Path(
    "/mnt/home/atanelus/ceph/datasets/adolescent_dataset_full.h5"
)
disbatch_script_path = base_dir / "run_experiment"


def get_command(
    pretrain_dataset_path: Optional[Path],
    base_config_path: Path,
):
    # Lines for loading the virtual environment
    lines = [
        "source ~/.bashrc",
        "source ~/venvs/general/bin/activate",
    ]
    if pretrain_dataset_path is not None:
        pt_save_path = (
            model_dir / f"{pretrain_dataset_path.stem}_{base_config_path.stem}_pretrain"
        )
        ft_save_path = (
            model_dir / f"{pretrain_dataset_path.stem}_{base_config_path.stem}_finetune"
        )
        log_name = f"{pretrain_dataset_path.stem}_{base_config_path.stem}.log"
        lines.extend(
            [
                f"python -u -m gerbilizer --config {base_config_path} --data {pretrain_dataset_path} --save-path {pt_save_path}",
                f"python -u -m gerbilizer --config {pt_save_path / 'config.json'} --data {finetune_dataset_path} --save-path {ft_save_path}",
            ]
        )
    else:
        ft_save_path = model_dir / f"{base_config_path.stem}_no_pretrain"
        log_name = f"{base_config_path.stem}_no_pretrain.log"
        lines.append(
            f"python -u -m gerbilizer --config {base_config_path} --data {finetune_dataset_path} --save-path {ft_save_path}",
        )
    return f'( {" && ".join(lines)} ) &> {log_dir / log_name}'


def make_experiment():
    with open(disbatch_script_path, "w") as f:
        for base_config_path in base_config_dir.iterdir():
            for pretrain_dataset_path in pretrain_dataset_paths:
                cmd = get_command(pretrain_dataset_path, base_config_path)
                f.write(cmd + "\n")
            cmd = get_command(None, base_config_path)
            f.write(cmd + "\n")

    print(f"Disbatch script written to {disbatch_script_path}")
    print("To run:")

    num_configs = len(list(base_config_dir.iterdir()))
    num_datasets = len(pretrain_dataset_paths)
    num_models = num_configs * num_datasets + num_configs
    num_jobs = min(20, num_models)  # Use at most 20 tasks
    print(
        "module load disBatch; "
        "mkdir disbatch_logs; "
        f"sbatch -n {num_jobs} -p gpu --gpus-per-task=1 -t 0-12 --mem=32GB -c 6 disBatch -p disbatch_logs/ {disbatch_script_path}"
    )


make_experiment()
