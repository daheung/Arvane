import os
import sys
import queue
import atexit
import asyncio
import numpy as np

from functools import partial
from concurrent.futures import  ThreadPoolExecutor, as_completed

from numpy.typing import NDArray
from typing import Tuple, Callable, Any

class InferenceThreadExecutor:
    def __init__(self):
        self.num_cpus = os.cpu_count() or 1
        self.cur_cpus = 0
        self.executor = ThreadPoolExecutor(max_workers=self.num_cpus)
        self._lock = asyncio.Lock()
        
        atexit.register(self.executor.shutdown, wait=True, cancel_futures=True)
    
    async def execute(self, task: Callable, *args, **kwargs):
        loop = asyncio.get_running_loop()
        fn = partial(task, *args, **kwargs)

        async with self._lock: 
            self.cur_cpus += 1

        try:
            return await loop.run_in_executor(self.executor, fn)
        finally:
            async with self._lock: 
                self.cur_cpus -= 1

    @property
    def max_workers(self) -> int:
        return self.num_cpus
    
    @property
    def num_workers(self) -> int:
        return self.cur_cpus