import os
import atexit
import asyncio
import threading
import logging

from functools import partial
from typing import Any, Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor


class InferenceThreadExecutor:
    def __init__(self, max_workers: int | None = None):
        if max_workers is None:
            max_workers = max(1, (os.cpu_count() or 4) - 1)

        self._max_workers = max_workers

        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="inference",
        )

        self._lock = threading.Lock()
        self._pending_tasks = 0

        # asyncio.Task 레지스트리
        self._tasks: dict[str, asyncio.Task[Any]] = {}

        atexit.register(self.shutdown)

    async def execute(
        self,
        task: Callable[..., Any],
        *args,
        **kwargs,
    ) -> Any:
        loop = asyncio.get_running_loop()
        fn = partial(task, *args, **kwargs)

        with self._lock:
            self._pending_tasks += 1

        try:
            return await loop.run_in_executor(
                self._executor,
                fn,
            )
        finally:
            with self._lock:
                self._pending_tasks -= 1

    def submit(
        self,
        task_id: str,
        task: Callable[..., Any],
        *args,
        **kwargs,
    ) -> asyncio.Task[Any]:
        """
        동기 함수를 thread pool에 제출하고,
        asyncio.Task를 내부 registry에 저장한다.

        실행 중인 event loop 내부에서 호출해야 한다.
        """
        existing_task = self._tasks.get(task_id)

        if existing_task is not None and not existing_task.done():
            raise RuntimeError(
                f"Task is already running: task_id={task_id}"
            )

        async_task = asyncio.create_task(
            self._run_registered_task(
                task_id,
                task,
                *args,
                **kwargs,
            ),
            name=f"inference:{task_id}",
        )

        self._tasks[task_id] = async_task
        return async_task

    async def _run_registered_task(
        self,
        task_id: str,
        task: Callable[..., Any],
        *args,
        **kwargs,
    ) -> Any:
        try:
            result = await self.execute(
                task,
                *args,
                **kwargs,
            )

            logging.info(
                "Inference task completed: task_id=%s",
                task_id,
            )

            return result

        except asyncio.CancelledError:
            logging.warning(
                "Inference task cancelled: task_id=%s",
                task_id,
            )
            raise

        except Exception:
            logging.exception(
                "Inference task failed: task_id=%s",
                task_id,
            )
            raise

        finally:
            # 현재 등록된 task가 자기 자신인 경우에만 제거
            current_task = asyncio.current_task()

            if self._tasks.get(task_id) is current_task:
                self._tasks.pop(task_id, None)

    def is_running(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        return task is not None and not task.done()

    def get_task(
        self,
        task_id: str,
    ) -> asyncio.Task[Any] | None:
        return self._tasks.get(task_id)

    def get_status(self, task_id: str) -> str:
        task = self._tasks.get(task_id)

        if task is None:
            return "not_found"

        if task.cancelled():
            return "cancelled"

        if not task.done():
            return "running"

        if task.exception() is not None:
            return "failed"

        return "completed"

    def cancel(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)

        if task is None or task.done():
            return False

        return task.cancel()

    @property
    def max_workers(self) -> int:
        return self._max_workers

    @property
    def num_pending_tasks(self) -> int:
        with self._lock:
            return self._pending_tasks

    @property
    def num_registered_tasks(self) -> int:
        return sum(
            1
            for task in self._tasks.values()
            if not task.done()
        )

    def shutdown(self) -> None:
        self._executor.shutdown(
            wait=True,
            cancel_futures=True,
        )