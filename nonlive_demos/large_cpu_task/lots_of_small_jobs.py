"""Wraps small_cpu_job.py to run a ton of them in parallel."""

from pathlib import Path

command_format = "date '+%Y-%m-%d %H:%M:%S.%6N'; source /mnt/home/atanelus/venvs/new/bin/activate; python {script_path}; date '+%Y-%m-%d %H:%M:%S.%6N'"


if __name__ == "__main__":
    cur_dir = Path(__file__).parent
    script_path = cur_dir / "small_cpu_job.py"
    log_dir = cur_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    with open(cur_dir / "disbatch_script", "w") as ctx:
        for i in range(40):
            # Runs the small_cpu_job.py script 40 times in parallel
            cmd = command_format.format(script_path=script_path)
            cmd = f"({cmd}) &> {log_dir}/job_{i:03d}.log"
            ctx.write(cmd)
            ctx.write("\n")

    print(f"Disbatch script written to {cur_dir / 'disbatch_script'}")
