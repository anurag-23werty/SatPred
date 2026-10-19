"""Process-parallel map for per-channel work (torch-safe)."""

import multiprocessing
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any


def parallel_map[T](fn: Callable[[Any], T], items: Sequence[Any], jobs: int = 1) -> Iterator[T]:
    """``fn`` over ``items``, results in input order, in ``jobs`` spawned worker processes.

    ``fn`` must be picklable (a top-level function or a ``functools.partial`` of one).
    Spawn rather than fork: forking a process that already started torch's thread pools can
    deadlock.
    """
    if jobs < 1:
        raise ValueError("jobs must be >= 1")
    if jobs == 1:
        yield from map(fn, items)
        return
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as pool:
        for future in [pool.submit(fn, item) for item in items]:
            yield future.result()
