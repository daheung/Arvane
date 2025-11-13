import os
import sys
import queue
import atexit
import asyncio
import numpy as np

from typing import Callable
from functools import partial
from concurrent.futures import  ThreadPoolExecutor

import threading

class InferenceThreadExecutor:
    def __init__(self, max_workers: int | None = None):
        if max_workers is None:
            max_workers = max(1, (os.cpu_count() or 4) - 1)

        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self._cur_cpus = 0

        atexit.register(self.executor.shutdown, wait=True, cancel_futures=True)

    async def execute(self, task: Callable, *args, **kwargs):
        loop = asyncio.get_running_loop()
        fn = partial(task, *args, **kwargs)

        with self._lock:
            self._cur_cpus += 1

        try:
            return await loop.run_in_executor(self.executor, fn)
        finally:
            with self._lock:
                self._cur_cpus -= 1

    @property
    def max_workers(self) -> int:
        return self.executor._max_workers

    @property
    def num_workers(self) -> int:
        return self._cur_cpus
