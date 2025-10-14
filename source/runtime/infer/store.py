import sys
import torch
import logging
import uuid as tid
import uuid as uid

from typing import Any, Dict

from source.runtime.infer.defines import InferenceStore, EStoreOperatorType, EStoreObjectType
from source.runtime.array import ChunkArrayConcurrent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logging.info("Loading InferenceChunkStoreConcurrent...")

# InferenceChunkStoreConcurrent 클래스
# TODO: 모델이 반환한 결과를 유저를 기준으로 분기하여 저장합니다. 
# 현재는 Map Container 방식입니다. 복잡도 O(1)
# 아직은 dict에 대한 원자적 연산, 멀티 스레드로 인한 동기화가 필요하지 않아 dict 전용 클래스를 따로 구현하지 않은 상태입니다.
#   => 필요하다면 그 때, 따로 구현하십시오.
#
# store_per_user: Dict[task_id, InferenceStore]로 구성되어 task_id를 key로 가집니다.
class InferenceChunkStoreConcurrent:
    def __init__(self, chunk_size=64):
        logging.info(f"InferenceChunkStoreConcurrent initialized. Chunk size: {chunk_size}")

        self._chunk_size = chunk_size
        self._items: Dict[str, InferenceStore] = dict()
        

    def add_task(self, user_id: str) -> str:
        task_id = tid.uuid1()
        self._items[task_id] = InferenceStore(
            user_id=user_id,
            task_id=task_id,
            chunk_size=self._chunk_size,
        )

        return task_id 
    
    def pop_task(self, task_id: str) -> InferenceStore:
        if not self._check_task_id(task_id):
            raise KeyError(f"Cannot find task_id: current task id: {task_id}")

        return self._items.pop(task_id)

    def update_object_by_task_id(
        self, 
        task_id: str, 
        objects: Dict[str, tuple[Any]], 
        tsk_type: EStoreObjectType, 
        op_type: EStoreOperatorType=EStoreOperatorType.INSERT
    ) -> int:
        if not self._check_task_id(task_id):
            raise KeyError(f"Cannot find task_id: current task id: {task_id}")
        
        keys = objects.keys()
        if (keys is None) or (len(keys) == 0):
            raise ValueError(f"Cannot update empty object: current task id: {task_id}")
        
        object_len_per_keys = [len(object) for object in objects]
        if any(length == 0 for length in object_len_per_keys) or \
           any(length != object_len_per_keys[0] for length in object_len_per_keys):
            raise ValueError(f"Cannot update object with different length: current task id: {task_id}, object keys: {keys}, object lengths: {object_len_per_keys}")

        store = self._items[task_id]
        if tsk_type & EStoreObjectType.DEPTH   : store.depth_container  .add_objects(objects.get("depth"  , None), op_type)
        if tsk_type & EStoreObjectType.POSE    : store.pose_container   .add_objects(objects.get("pose"   , None), op_type)
        if tsk_type & EStoreObjectType.IMAGE   : store.image_container  .add_objects(objects.get("image"  , None), op_type)
        if tsk_type & EStoreObjectType.K_IMAGE : store.k_image_container.add_objects(objects.get("k_image", None), op_type)
        if tsk_type & EStoreObjectType.K_DEPTH : store.k_depth_container.add_objects(objects.get("k_depth", None), op_type)

        return len(objects)

    def load_object_by_task_id(self, task_id: str, tsk_type: EStoreObjectType) -> Dict[str, ChunkArrayConcurrent]:
        if not self._check_task_id(task_id):
            raise KeyError(f"Cannot find task_id: current task id: {task_id}")

        store = self._items[task_id]
        ret_container = dict()

        if tsk_type & EStoreObjectType.DEPTH  : ret_container["depth"  ] = store.depth_container
        if tsk_type & EStoreObjectType.POSE   : ret_container["pose"   ] = store.pose_container
        if tsk_type & EStoreObjectType.IMAGE  : ret_container["image"  ] = store.image_container
        if tsk_type & EStoreObjectType.K_IMAGE: ret_container["k_image"] = store.k_image_container
        if tsk_type & EStoreObjectType.K_DEPTH: ret_container["k_depth"] = store.k_depth_container

        return ret_container

    def exists(self, task_id: str) -> bool:
        return self._check_task_id(task_id)
    
    def _check_task_id(self, task_id: str) -> bool:
        return task_id in self._items.keys()