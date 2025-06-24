import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_runtimes(runtimes: np.ndarray, title="Parallelized Small CPU Jobs"):
    """
    Visualized the runtimes of parallelized small CPU jobs.
    Args:
        runtimes (np.ndarray): (n, 2) array of start and end times of each job.
        title (str, optional): Figure title. Defaults to "Parallelized Small CPU Jobs".
    """

    # Although the jobs are started in order, the time taken to initialize them (loading modules, venv, etc.)
    # can differ, causing the start time of the script to be slightly out of order.
    sorting = runtimes[:, 0].argsort()
    runtimes = runtimes[sorting, :]

    runtimes -= runtimes[0, 0]  # Normalize start times to zero

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        y=np.arange(len(runtimes)),
        width=runtimes[:, 1] - runtimes[:, 0],
        left=runtimes[:, 0],
        color="skyblue",
        align="edge",
    )
    ax.set_ylim(len(runtimes) + 0.5, -0.5)  # Reverse y-axis for better readability
    ax.set_xlabel("Wall time (seconds)")
    ax.set_ylabel("Job index (sorted by start time)")
    ax.set_title(title)

    plt.savefig("parallelized_small_cpu_jobs_runtimes.png", dpi=300)


def get_runtimes_from_logs(log_dir: Path):
    """
    Extracts runtimes from log files in the specified directory.
    Args:
        log_dir (str): Directory containing the log files.
    Returns:
        np.ndarray: Array of start and end times for each job.
    """
    runtimes = []
    for log_path in sorted(log_dir.glob("job_*.log")):
        start_time, end_time = None, None
        with open(log_path, "r") as f:
            lines = f.readlines()
            dt_fmt = "%Y-%m-%d %H:%M:%S.%f"
            try:
                start_time = datetime.strptime(lines[0].strip(), dt_fmt).timestamp()
                end_time = datetime.strptime(lines[-1].strip(), dt_fmt).timestamp()
            except:
                continue
        runtimes.append([start_time, end_time])
    return np.array(runtimes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot runtimes of parallelized small CPU jobs."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory containing the log files.",
    )
    args = parser.parse_args()

    runtimes = get_runtimes_from_logs(args.log_dir)
    if runtimes.size == 0:
        print("No valid runtimes found in the log directory.")
    else:
        plot_runtimes(runtimes)
