import sys
import time
import logging
import threading

from dataclasses import dataclass
from typing import Any, List, Sequence, Callable, Generic, TypeVar, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logging.info("Loading ChunkArrayConcurrent...")

T = TypeVar("T")

@dataclass(frozen=True)
class FixedChunkObject(Generic[T]):
    key: Optional[float]
    object: T

class FixedChunkArray(Generic[T]):
    def __init__(self, chunk_size):
        self._items: List[FixedChunkObject[T]] = list()
        self._items_lock = threading.RLock()
        self._chunk_size: int = int(chunk_size)

    def __len__(self):
        with self._items_lock:
            return len(self._items)
    
    def __getitem__(self, idx) -> FixedChunkObject[T]:
        if not (self._check_idx(idx)):
            raise IndexError("index out of range")
        
        with self._items_lock:
            return self._items[idx]
    
    @property
    def chunk_size(self):
        return self._chunk_size

    @property
    def num_item(self):
        with self._items_lock:
            return len(self._items)

    def add_object(self, object: T, key: Optional[float] = None) -> int:
        assert not (self._is_full())

        with self._items_lock:
            entry = FixedChunkObject(key=key,
                               object=object)
            self._items.append(entry)
        
        return self.num_item - 1

    
    def add_objects(self, objects: Sequence[T], keys: Optional[Sequence[float]] = None) -> int:
        """
        여러 객체를 한꺼번에 추가합니다. 기본적으로 다음에 넣어야할 배열의 인덱스가 반환됩니다.
        추가 후 배열이 다 차서 다음에 추가로 넣을 공간이 없다면 -1을 반환합니다.
        """
        assert len(objects) > 0
        assert len(objects) + len(self._items) <= self._chunk_size

        with self._items_lock:
            for idx, object in enumerate(objects):
                assert idx < len(keys) if keys is not None else True
                key = keys[idx] if keys is not None else None
                entry = FixedChunkObject(key=key,
                                   object=object)
                
                self._items.append(entry)
        
        if (self._is_full()):
            return -1
        
        return self.num_item

    def _insert_considering_chunk(self, idx: int, object: T, key: Optional[float] = None) -> Optional[FixedChunkObject[T]]:
            """
            지정된 인덱스에 요소를 삽입합니다.
            - 공간이 있으면: 그냥 삽입하고 None 반환
            - 꽉 찼으면: 삽입 후 가장 마지막 요소를 제거(pop)하여 반환 (오버플로우 처리)
            """
            with self._items_lock:
                # 인덱스 보정 (append 처리를 위해)
                insert_idx = min(idx, len(self._items))
                
                entry = FixedChunkObject(key=key, object=object)
                self._items.insert(insert_idx, entry)
                
                overflow = None
                if len(self._items) > self._chunk_size:
                    # 꽉 찼으므로 마지막 요소를 뺌
                    overflow = self._items.pop()
                    
                return overflow
        
    def delete_object(self, predicate: Callable[[FixedChunkObject, int], bool]) -> int:
        return self._delete_object_by_predicate(predicate=predicate)
    
    def _delete_object_by_predicate(self, predicate: Callable[[FixedChunkObject, int], bool]) -> int:
        old_len = len(self._items)
        with self._items_lock:    
            self._items = [item for i, item in enumerate(self._items) if not predicate(item, i)]

        return old_len - len(self._items)
        
    def _check_idx(self, idx):
        return 0 <= idx <= self._chunk_size - 1
    
    def _is_full(self):
        return len(self) >= self._chunk_size
    
class ChunkArrayConcurrent(Generic[T]):
    """
    - 전역 배열 = [FixedChunkArray, FixedChunkArray, ...] 의 나열
    - 길이 변화(append, 새 청크 생성)는 self._chunks_lock 로 보호
    - 개별 요소 접근/수정은 해당 청크의 락으로만 보호 (락의 범위를 좁힘)
    - 전역 인덱스 -> (청크ID, 오프셋) 매핑
    """
    def __init__(self, chunk_size: int = 64):
        self._chunk_size = int(chunk_size)
        self._chunks: List[FixedChunkArray[T]] = []
        self._chunks_lock = threading.Lock()
        
    # ---- 프로퍼티 ----
    @property
    def chunk_size(self) -> int:
        return self._chunk_size
    
    @property
    def num_chunk(self) -> int:
        with self._chunks_lock:
            return len(self._chunks)
    
    @property
    def num_capacity(self) -> int:
        with self._chunks_lock:
            return len(self._chunks) * self._chunk_size
    
    def __getitem__(self, index: int):
        return self.get(index)

    def __len__(self) -> int:
        # 총 아이템 수 = 각 청크 길이의 합 (읽기 전용이므로 청크별 락만 사용)
        with self._chunks_lock:
            total = 0
            for ch in self._chunks:
                total += len(ch)

            return total

    # ---- 내부 유틸: 전역 인덱스 <-> (청크ID, 오프셋) ----
    def _global_to_local(self, idx: int) -> tuple[int, int]:
        """전역 idx → (chunk_idx, offset)"""
        if idx < 0:
            raise IndexError("index out of range")
        
        chunk_idx = idx // self._chunk_size
        offset = idx % self._chunk_size
        with self._chunks_lock:
            if chunk_idx >= len(self._chunks):
                raise IndexError("index out of range")
            
        return chunk_idx, offset

    def _ensure_chunk_exists_for_append(self) -> FixedChunkArray[T]:
        """append를 위한 마지막 청크 확보(가득 찼으면 새 청크 생성)."""
        with self._chunks_lock:
            if not self._chunks or len(self._chunks[-1]) >= self._chunk_size:
                self._chunks.append(FixedChunkArray[T](self._chunk_size))
            return self._chunks[-1]

    def _get_chunk(self, chunk_idx: int) -> FixedChunkArray[T]:
        with self._chunks_lock:
            if chunk_idx < 0 or chunk_idx >= len(self._chunks):
                raise IndexError("chunk id out of range")
            
            return self._chunks[chunk_idx]

    # ---- 배열 API ----
    def add_object(self, object: T, key: Optional[float] = None) -> int:
        """
        새 객체를 마지막 청크에 추가. 가득 차 있으면 새 청크를 생성.
        """
        chunk = self._ensure_chunk_exists_for_append()
        # 청크 자체가 락을 관리하므로 여기서는 청크 레벨만 사용
        offset = chunk.add_object(object, key=key)
        chunk_idx = len(self._chunks) - 1

        return (chunk_idx * self.chunk_size) + offset

    def add_objects(self, objects: Sequence[T], keys: Optional[Sequence[float]] = None) -> int:
        objects_len = len(objects)
        object_offset = 0  # 전역 진행 포인터

        while object_offset < objects_len:
            # 마지막 청크를 확보(가득 차 있으면 내부에서 새 청크 보장한다고 가정)
            chunk = self._ensure_chunk_exists_for_append()

            free = self._chunk_size - chunk.num_item
            if free <= 0:
                # 방어적: 혹시라도 보장이 안 됐을 때 다시 확보
                chunk = self._ensure_chunk_exists_for_append()
                free = self._chunk_size - chunk.num_item
                if free <= 0:
                    # 이 경우는 구현 계약 위반이므로 명확히 실패
                    raise RuntimeError("No free space even after ensuring chunk for append")

            remaining = objects_len - object_offset
            to_add = min(remaining, free)

            # 이번 턴에 넣을 슬라이스
            batch = objects[object_offset : object_offset + to_add]

            # add_objects의 반환 값은 "오프셋"입니다.
            keys = keys[object_offset : object_offset + to_add] if keys is not None else None
            chunk_offset = chunk.add_objects(batch, keys=keys)
            object_offset += to_add

            # 혹시 아무 것도 못 넣었다면(비정상) 무한루프 방지
            if chunk_offset == 0:
                raise RuntimeError("chunk.add_objects didn't add any items; aborting to avoid infinite loop")

        return object_offset  # 실제 추가된 총 개수


    def get(self, idx: int) -> FixedChunkObject[T] | None:
        """
        전역 인덱스에서 항목을 읽습니다.
        """
        chunk_idx, offset = self._global_to_local(idx)
        chunk = self._get_chunk(chunk_idx)
        if not chunk._check_idx(offset):
            return None

        return chunk[offset]

    def set(self, idx: int, new_object: T, new_key: Optional[float] = None) -> FixedChunkObject[T]:
        """
        전역 인덱스의 항목을 새 object로 교체.
        FixedChunkArray에 set API가 없으므로, 안전하게 락을 잡고 내부 리스트를 교체합니다.
        """
        chunk_idx, offset = self._global_to_local(idx)
        chunk = self._get_chunk(chunk_idx)

        # 청크 내부 교체(락 포함)
        with chunk._items_lock:
            if offset >= len(chunk._items):
                raise IndexError("index out of range")
            old = chunk._items[offset]
            updated = FixedChunkObject[T](
                key=new_key,
                object=new_object
            )
            chunk._items[offset] = updated

        return updated
    
    def get_raw_objects(
        self,
        start: int =  0, 
        size : int = -1
    ) -> List[T]:
        objects = self.get_objects(start=start, size=size)
        return [object.object for object in objects]

    def get_objects(
        self, 
        start: int =  0, 
        size : int = -1
    ) -> Sequence[FixedChunkObject[T]]:
        """
        [start, start+size) 범위를 반환(읽기).
        여러 청크에 걸칠 수 있으므로, 범위를 청크 단위로 분할해 각 청크에서 안전하게 읽습니다.
        """
        size = len(self) - start if size < 0 else size
        end = start + size
        n = len(self)
        if not (0 <= start <= end <= n):
            raise IndexError("range out of bounds")
        if size == 0:
            return []

        out: List[FixedChunkObject[T]] = []
        # 범위를 청크 경계로 나눠 순서대로 수집
        cur = start
        while cur < end:
            chunk_idx, offset = self._global_to_local(cur)
            chunk = self._get_chunk(chunk_idx)
            # 해당 청크에서 읽을 수 있는 최대 길이
            take = min(end - cur, self._chunk_size - offset)
            with chunk._items_lock:
                # 청크 길이가 줄었을 수 있으니 재확인
                upper = min(offset + take, len(chunk._items))
                # 보기용 전역 ID로 변환
                for j in range(offset, upper):
                    item = chunk._items[j]
                    out.append(
                        FixedChunkObject[T](key=item.key, object=item.object)
                    )
            cur += take
        return out

    def set_objects(self, start: int, values: Sequence[T], keys: Optional[Sequence[float]] = None) -> None:
        """
        [start, start+len(values)) 범위에 대해 값들을 교체(쓰기).
        범위가 여러 청크에 걸쳐도 청크별로 잠그고 교체합니다.
        """
        if not values:
            return
        end = start + len(values)
        n = len(self)
        if not (0 <= start < end <= n):
            raise IndexError("range out of bounds")

        cur = start
        k = 0
        while cur < end:
            chunk_idx, offset = self._global_to_local(cur)
            ch = self._get_chunk(chunk_idx)
            take = min(end - cur, self._chunk_size - offset)
            with ch._items_lock:
                upper = min(offset + take, len(ch._items))
                span = upper - offset
                for j in range(span):
                    assert k + j < len(keys) if keys is not None else True
                    key = keys[k + j] if keys is not None else None
                    ch._items[offset + j] = FixedChunkObject[T](
                        key=key,
                        object=values[k + j]
                    )
            cur += take
            k += take

    def insert(self, idx: int, object: T, key: Optional[float] = None) -> None:
            """
            특정 위치(idx)에 객체를 삽입합니다.
            해당 위치의 청크부터 뒤쪽 청크들로 데이터가 하나씩 밀려나는(Shift) 비용이 발생합니다.
            """
            # 구조적 변경(새 청크 추가 가능성)이 있으므로 chunks_lock을 잡습니다.
            with self._chunks_lock:
                total_len = 0
                for ch in self._chunks:
                    total_len += len(ch)
                
                if idx < 0:
                    raise IndexError("index cannot be negative")
                if idx > total_len:
                    raise IndexError("index out of range")
    
                # 1. 삽입 시작 위치 계산
                start_chunk_idx = idx // self._chunk_size
                start_offset = idx % self._chunk_size
                
                # 만약 맨 뒤에 추가하는 경우(append와 동일)이고, 
                # 계산된 chunk_idx가 현재 청크 개수와 같다면(새 청크 필요) 처리
                if start_chunk_idx >= len(self._chunks):
                    # 마지막 청크가 꽉 차서 다음 청크 인덱스를 가리키거나, 배열이 비어있는 경우
                    new_chunk = FixedChunkArray(self._chunk_size)
                    new_chunk.add_object(object, key=key)
                    self._chunks.append(new_chunk)
                    return
    
                # 2. Ripple Shift (도미노 이동)
                # 현재 청크에 삽입 -> 넘치는 것을 다음 청크의 맨 앞에 삽입 -> 반복
                current_obj_to_insert = object
                
                # 시작 청크부터 마지막 청크까지 순회
                for i in range(start_chunk_idx, len(self._chunks)):
                    chunk = self._chunks[i]
                    
                    # 첫 번째(타겟) 청크는 계산된 offset에 넣고,
                    # 그 이후 청크들은 앞쪽에서 밀려온 것이므로 0번 인덱스에 넣습니다.
                    local_insert_idx = start_offset if i == start_chunk_idx else 0
                    
                    # 삽입 시도 및 오버플로우(밀려난 녀석) 획득
                    overflow_obj: FixedChunkObject[T] = chunk._insert_considering_chunk(local_insert_idx, current_obj_to_insert, key=key)
                    
                    if overflow_obj is None:
                        # 청크에 빈 공간이 있어서 밀려난 게 없음 -> 연쇄 이동 종료
                        return
                    else:
                        # 밀려난 요소를 다음 청크의 입력으로 설정
                        current_obj_to_insert = overflow_obj.object
                        current_obj_to_key = overflow_obj.key
    
                # 3. 마지막 청크까지 처리했는데도 남은 요소(overflow)가 있다면 새 청크 생성
                # (위 루프가 끝났다는 건 마지막 청크에서도 하나가 튕겨져 나왔다는 뜻)
                new_chunk = FixedChunkArray(self._chunk_size)
                new_chunk.add_object(current_obj_to_insert, key=current_obj_to_key)
                self._chunks.append(new_chunk)