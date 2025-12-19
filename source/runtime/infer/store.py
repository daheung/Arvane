import sys
import torch
import logging
import uuid as tid
import uuid as uid

from typing import Any, Dict, Tuple, Callable, TypeVar, Optional

from source.runtime.container.array import ChunkArrayConcurrent
from source.runtime.infer.defines import (
    InferenceStore, 
    EStoreOperatorType, 
    EStoreObjectType, 
    ReconLog,
    StoreStatus,
)

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
    
    def __getitem__(self, task_id: str) -> InferenceStore:
        if not self._check_task_id(task_id):
            raise KeyError(f"Cannot find task_id: current task id: {task_id}")

        return self._items[task_id]
    
    def add_task(self, user_id: str) -> str:
        task_id: str = str(tid.uuid1())
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
        timestamp: Optional[float],
        tsk_type: EStoreObjectType,
    ) -> int:
        if not self._check_task_id(task_id):
            raise KeyError(f"Cannot find task_id: current task id: {task_id}")
        
        keys = objects.keys()
        if (keys is None) or (len(keys) == 0):
            raise ValueError(f"Cannot update empty object: current task id: {task_id}")
        
        # depth의 경우에는 나중에 모델에서 처리, 해당 이유로 depth는 체크하지 않음
        object_len_per_keys = [len(item) for (key, item) in objects.items() if key != "depth"]
        if any(length == 0 for length in object_len_per_keys) or \
           any(length != object_len_per_keys[0] for length in object_len_per_keys):
            raise ValueError(f"Cannot update object with different length: current task id: {task_id}, object keys: {keys}, object lengths: {object_len_per_keys}")

        store = self._items[task_id]
        depths   = objects.get("depth"  , None)
        poses    = objects.get("pose"   , None)
        images   = objects.get("image"  , None)
        k_images = objects.get("k_image", None)
        k_depths = objects.get("k_depth", None)
        f_px     = objects.get("f_px"   , None)

        if tsk_type & EStoreObjectType.DEPTH       : store.depth_container       .add_objects(depths  , [timestamp for _ in range(len(depths  ))] if timestamp is not None else None)
        if tsk_type & EStoreObjectType.POSE        : store.pose_container        .add_objects(poses   , [timestamp for _ in range(len(poses   ))] if timestamp is not None else None)
        if tsk_type & EStoreObjectType.IMAGE       : store.image_container       .add_objects(images  , [timestamp for _ in range(len(images  ))] if timestamp is not None else None)
        if tsk_type & EStoreObjectType.K_IMAGE     : store.k_image_container     .add_objects(k_images, [timestamp for _ in range(len(k_images))] if timestamp is not None else None)
        if tsk_type & EStoreObjectType.K_DEPTH     : store.k_depth_container     .add_objects(k_depths, [timestamp for _ in range(len(k_depths))] if timestamp is not None else None)
        if tsk_type & EStoreObjectType.FOCAL_LENGTH: store.focal_length_container.add_objects(f_px    , [timestamp for _ in range(len(f_px    ))] if timestamp is not None else None)

        return 1

    def load_object_by_task_id(self, task_id: str, tsk_type: EStoreObjectType) -> Dict[str, ChunkArrayConcurrent | ReconLog]:
        if not self._check_task_id(task_id):
            raise KeyError(f"Cannot find task_id: current task id: {task_id}")

        FIELD_MAP: dict[EStoreObjectType, Tuple[str, str]] = {
            EStoreObjectType.USER_ID      : ("user_id",      "user_id"),
            EStoreObjectType.TASK_ID      : ("task_id",      "task_id"),
            EStoreObjectType.IMAGE        : ("image",        "image_container"),
            EStoreObjectType.DEPTH        : ("depth",        "depth_container"),
            EStoreObjectType.POSE         : ("pose",         "pose_container"),
            EStoreObjectType.K_IMAGE      : ("k_image",      "k_image_container"),
            EStoreObjectType.K_DEPTH      : ("k_depth",      "k_depth_container"),
            EStoreObjectType.FOCAL_LENGTH : ("f_px",         "focal_length_container"),
            EStoreObjectType.RECON_STORE  : ("recon_con",    "recon_container"),
            EStoreObjectType.RECON_LOG    : ("recon_log",    "recon_log"),
            EStoreObjectType.RECON_RESULT : ("recon_result", "recon_object"),
            EStoreObjectType.RECON_STATUS : ("recon_status", "store_status"),
            EStoreObjectType.EXTRACTION_OBJECT : ("extraction_object", "extraction_object"),
        }

        store = self._items[task_id]
        ret_container = dict()
        for flag, (key, attr_name) in FIELD_MAP.items():
            if tsk_type & flag:
                ret_container[key] = getattr(store, attr_name)

        return ret_container

    def exists(self, task_id: str) -> bool:
        return self._check_task_id(task_id)
    
    def _check_task_id(self, task_id: str) -> bool:
        return task_id in self._items.keys()