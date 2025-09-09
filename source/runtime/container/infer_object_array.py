import time
from enum import Enum
from dataclasses import dataclass
from typing import Any, List, Sequence, Optional, Callable

import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, List, Sequence, Iterable

# ---------- 간단한 RWLock (stdlib에는 없어 직접 구현) ----------
class RWLock:
    def __init__(self):
        self._readers = 0
        self._r_lock = threading.Lock()
        self._w_lock = threading.Lock()
        self._r_ok = threading.Condition(self._r_lock)

    @contextmanager
    def read_lock(self):
        with self._r_lock:
            # writer가 잡고 있으면 대기
            while self._w_lock.locked():
                self._r_ok.wait()
            self._readers += 1
            
        try:
            yield
        finally:
            with self._r_lock:
                self._readers -= 1
                if self._readers == 0:
                    # 마지막 리더가 나가면 대기중인 writer에게 기회
                    self._r_ok.notify_all()

    @contextmanager
    def write_lock(self):
        # writer 단독 진입 보장
        with self._w_lock:
            # 리더가 모두 빠질 때까지 기다림
            with self._r_lock:
                while self._readers > 0:
                    self._r_ok.wait()
            try:
                yield
            finally:
                with self._r_lock:
                    self._r_ok.notify_all()

# ---------- 데이터 ----------
@dataclass(frozen=True)
class DepthEntry:
    ts_sec: float
    internal_id: int
    depth_object: Any

class FixedChunkArray:
    def __init__(self, chunk_size):
        self._items: List[DepthEntry] = list()
        self._items_lock = threading.Lock()
        self._chunk_size: int = int(chunk_size)
        self._next_internal_id: int = int(0)

    def __len__(self):
        with self._items_lock:
            return len(self._items)
    
    def __getitem__(self, idx):
        if not (self._check_idx(idx)):
            return None
        
        with self._items_lock:
            return self._items[idx]
    
    @property
    def chunk_size(self):
        return self._chunk_size

    @property
    def num_item(self):
        return len(self._items)

    def add_object(self, depth):
        if (self._is_full()):
            return
        
        with self._items_lock:
            entry = DepthEntry(ts_sec=time.time(),
                               internal_id=len(self._items),
                               depth_object=depth)
            self._items.append(entry)

    def delete_object(self, predicate: Callable[[DepthEntry, int], bool]) -> int:
        return self._delete_object_by_predicate(predicate=predicate)
    
    def _delete_object_by_predicate(self, predicate: Callable[[DepthEntry, int], bool]) -> int:
        old_len = len(self._items)
        with self._items_lock:    
            self._items = [item for i, item in enumerate(self._items) if not predicate(item, i)]

        return old_len - len(self._items)
        
    def _check_idx(self, idx):
        return 0 <= idx <= self._chunk_size - 1
    
    def _is_full(self):
        return len(self) >= self._chunk_size

# ---------- 부분 동기화 가능한 배열 ----------
class DepthObjectArrayConcurrent:
    def __init__(self, chunk_size: int = 64):
        self._items: List[DepthEntry] = []
        self._items_lock = threading.Lock()  # 크기 변화(append 등) 보호
        self._chunk_size = int(chunk_size)
        self._locks: List[RWLock] = []       # 청크별 RWLock (동적 확장)

    @property
    def chunk_size(self):
        return self.chunk_size
    
    @property
    def num_chunk(self):
        return 
    
    # 내부: 인덱스 -> 샤드 ID
    def _chunk_id(self, idx: int) -> int:
        return idx // self._chunk_size

    # 내부: 샤드 id 범위를 보장하며 락 배열 확장
    def _ensure_chunks_upto(self, chunk_id: int):
        with self._items_lock:
            while chunk_id >= len(self._locks):
                self._locks.append(RWLock())

    # 내부: 범위에 필요한 샤드 id들
    def _range_chunks(self, start: int, end_exclusive: int) -> Sequence[int]:
        if start < 0 or end_exclusive < start:
            raise IndexError("invalid range")
        if start == end_exclusive:
            return []
        s0 = self._chunk_id(start)
        s1 = self._chunk_id(end_exclusive - 1)
        # 샤드 락 배열이 충분히 있는지 보장
        self._ensure_chunks_upto(s1)
        return list(range(s0, s1 + 1))

    # 여러 샤드에 대해 읽기/쓰기 락을 "정렬 순서"로 획득하여 교착 방지
    @contextmanager
    def _lock_chunks(self, chunk_ids: Iterable[int], write: bool):
        chunk_ids = sorted(set(chunk_ids))
        contexts = []
        try:
            # 순서대로 락 획득
            for sid in chunk_ids:
                ctx = (self._locks[sid].write_lock() if write
                       else self._locks[sid].read_lock())
                ctx.__enter__()
                contexts.append(ctx)
            yield
        finally:
            # 역순으로 해제
            for ctx in reversed(contexts):
                ctx.__exit__(None, None, None)

    # -------- 배열 API --------
    def __len__(self) -> int:
        # 길이는 빈번히 읽히므로 가벼운 보호만
        with self._items_lock:
            return len(self._items)

    def add_object(self, depth: Any) -> DepthEntry:
        # append는 "크기 변화"이므로 전역 길이 락으로 보호
        with self._items_lock:
            entry = DepthEntry(ts_sec=time.time(),
                               internal_id=len(self._items),
                               depth_object=depth)
            self._items.append(entry)
            # 새로 늘어난 인덱스가 속한 샤드가 존재하도록 보장
            self._ensure_chunks_upto(self._chunk_id(len(self._items)-1))
            return entry

    def get(self, idx: int) -> DepthEntry:
        # 읽기: 해당 인덱스 샤드만 읽기 락
        n = len(self)  # 길이 스냅샷
        if not (0 <= idx < n):
            raise IndexError("index out of range")
        sid = self._chunk_id(idx)
        self._ensure_chunks_upto(sid)
        with self._lock_chunks([sid], write=False):
            # 길이 변동 가능성 최소화 위해 인덱스 유효성 재확인
            if idx >= len(self):
                raise IndexError("index changed during read")
            return self._items[idx]

    def set(self, idx: int, new_depth: Any) -> DepthEntry:
        # 쓰기: 해당 인덱스 샤드만 쓰기 락
        n = len(self)
        if not (0 <= idx < n):
            raise IndexError("index out of range")
        sid = self._chunk_id(idx)
        self._ensure_chunks_upto(sid)
        with self._lock_chunks([sid], write=True):
            old = self._items[idx]
            updated = DepthEntry(ts_sec=time.time(),
                                 internal_id=old.internal_id,
                                 depth_object=new_depth)
            self._items[idx] = updated
            return updated

    def get_chunk(self, start: int, size: int) -> Sequence[DepthEntry]:
        if size < 0:
            raise ValueError("size must be >= 0")
        end = start + size
        n = len(self)
        if not (0 <= start <= end <= n):
            raise IndexError("range out of bounds")
        chunk_ids = self._range_chunks(start, end)
        with self._lock_chunks(chunk_ids, write=False):
            # 경계 재확인 (동시 append 대비)
            n2 = len(self)
            if end > n2:
                end = n2
            return list(self._items[start:end])

    def set_chunk(self, start: int, values: Sequence[Any]) -> None:
        end = start + len(values)
        n = len(self)
        if not (0 <= start < end <= n):
            raise IndexError("range out of bounds")
        chunk_ids = self._range_chunks(start, end)
        with self._lock_chunks(chunk_ids, write=True):
            for off, v in enumerate(values):
                i = start + off
                old = self._items[i]
                self._items[i] = DepthEntry(ts_sec=time.time(),
                                            internal_id=old.internal_id,
                                            depth_object=v)
