""" Demonstration of converting a python script to a command line tool using argparse. """

import time
from pathlib import Path

import numpy as np


def expensive_function(a: int, save_path: Path):
    """A function that takes a while to run.
    Happens to be a great target for parallelization.
    """

    time.sleep(3)
    data = np.random.rand(a)
    output = " ".join(map(str, data))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as ctx:
        ctx.write(output)


if __name__ == "__main__":
    # expensive_function(10, Path("outputs/run_a.txt"))
    # expensive_function(11, Path("outputs/run_b.txt"))
    expensive_function(13, Path("outputs/run_c.txt"))
    # expensive_function(9, Path("outputs/run_d.txt"))

    # kinda inconvenient, no?
