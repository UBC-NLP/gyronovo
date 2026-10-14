import os
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path.cwd() / ".numba-cache"))

import multiprocessing as mp
from tqdm import tqdm


def _transform_workers() -> int:
    try:
        workers = int(os.getenv("NOVOBENCH_TRANSFORM_WORKERS", "4"))
    except ValueError:
        workers = 4
    return max(workers, 1)


def _map_transform(func, args_iterable):
    workers = _transform_workers()
    if workers <= 1:
        return [func(args) for args in tqdm(args_iterable)]
    with mp.Pool(processes=workers) as pool:
        return list(tqdm(pool.map(func, args_iterable)))
