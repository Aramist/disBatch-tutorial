"""This is a small CPU job that runs with a small amount of RAM and only requires 4 cores"""

import time
from pprint import pformat

import numpy as np
from joblib import Parallel, delayed


def job():
    # Simulate a small CPU task
    start_time = time.time()

    # Create a large array and perform some computations
    gen = np.random.default_rng(seed=42)
    rand_mat = gen.random((2000, 2000), dtype=np.float32)  # ~15MiB
    rand_vec = gen.random((2000,), dtype=np.float32)
    result = rand_mat @ rand_vec  # mvp

    end_time = time.time()
    return end_time - start_time


def main():
    results = Parallel(n_jobs=4)(delayed(job)() for _ in range(4))
    print(f"Runtimes: {pformat(results)}")


if __name__ == "__main__":
    main()
