import os
import sys
import time
import torch
import logging
import threading
import numpy as np

from functools import partial
from numpy.typing import NDArray
from fastapi import BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from typing import Any, Tuple, Dict, List, Optional, TypeVar, Generic, Callable

from source.utils import load_config
from source.predictor.recon.predictor import ReconPredictor 
from source.predictor.depth.predictor import DepthPredictor
from source.runtime.infer.store import InferenceChunkStoreConcurrent
from source.runtime.infer.defines import StoreStatus, EStoreObjectType, EStoreOperatorType
from source.engine.defines import TaskStatus
from source.runtime.container.array import ChunkArrayConcurrent
from source.runtime.executor.infer_task_executor import InferenceThreadExecutor

class ArvaneEngine:
    depth_predictor: DepthPredictor
    recon_predictor: ReconPredictor

    container: InferenceChunkStoreConcurrent
    
    proxy_executor: InferenceThreadExecutor

    def __init__(self):
        logging.info("Initializing ArvaneEngine ...")
        depth_config, recon_config = load_config()
    
        depth_predictor = DepthPredictor(depth_config)
        depth_predictor.init()

        recon_predictor = ReconPredictor(recon_config)
        recon_predictor.init()

        # set app state dependencies
        self.depth_predictor = depth_predictor
        self.recon_predictor = recon_predictor

        # load container
        self.container = InferenceChunkStoreConcurrent(chunk_size=64)

        # init proxy executor
        self.proxy_executor = InferenceThreadExecutor()

    async def run_process(self, task_id: str):
        try:
            self.container[task_id].store_status = StoreStatus.DEPTH
            depth_status: TaskStatus = await self.proxy_executor.execute(
                self._update_depth_object_by_task_id, 
                task_id
            )
            if (depth_status != TaskStatus.SUCCESS):
                logging.warning(f"Auto-update depth failed (bg) for task_id={task_id}, status={depth_status.name}")

            self.container[task_id].store_status = StoreStatus.RECON
            recon_status: TaskStatus = await self.proxy_executor.execute(
                self._inference_recon_impl, 
                task_id
            )
            if (recon_status != TaskStatus.SUCCESS):
                logging.warning(f"Auto-update reconstruction failed (bg) for task_id={task_id}, status={recon_status.name}")

            self.container[task_id].store_status = StoreStatus.DONE
            ...
        except Exception as e:
            self.container[task_id].store_status = StoreStatus.ABORTED
            logging.exception(f"Auto-update crashed (bg) for task_id={task_id}: {e}")

        finally:
            ...
            
    async def update_depth_object_by_task_id(self, task_id: str):
        try:
            status = await run_in_threadpool(self._update_depth_object_by_task_id, task_id)
            if status != TaskStatus.SUCCESS:
                logging.warning(f"Auto-update depth failed (bg) for task_id={task_id}, status={status.name}")
            else:
                logging.info(f"Auto-update depth done (bg) for task_id={task_id}")

        except Exception as e:
            logging.exception(f"Auto-update depth crashed (bg) for task_id={task_id}: {e}")
    
    def _update_depth_object_by_task_id(self, task_id: str) -> TaskStatus:
        if not self.container.exists(task_id):
            logging.warning(f"Task was aborted. Cannot find task_id: {task_id}, create new task_id before infer depth.")
            return TaskStatus.ABORTED.value
        
        store = self.container.load_object_by_task_id(task_id, EStoreObjectType.IMAGE | EStoreObjectType.DEPTH)
        image_container: Optional[ChunkArrayConcurrent] = store["image"]
        depth_container: Optional[ChunkArrayConcurrent] = store["depth"]

        assert (depth_container is not None, "Depth container is None, but must be initialized already.")

        if (image_container is None):
            logging.warning(f"Image data is empty. Cannot infer depth for task_id: {task_id}")
            return TaskStatus.ABORTED.value

        num_depth: int = len(depth_container)
        num_image: int = len(image_container)
        # case 1) depth_container == image_container:
        #  더 이상 할 작업이 없음.
        # case 2) depth_container >  image_container:
        #  작업상 depth_container가 image_container보다 더 클 수 없음;
        #  같이 이미지를 중복 복원 or 배열의 일부 부분이 비어있을 가능성 있음
        if (num_depth >= num_image):
            logging.info(f"Depth data already exists. Skip depth inference for task_id: {task_id}")
            return TaskStatus.NOT_MODIFIED.value

        offset_start = num_image - num_image
        for idx in range(offset_start, num_image):
            depth, f_px = self._inference_depth_and_f_px_impl(image_container[idx].object)

            self.container.update_object_by_task_id(
                task_id=task_id,
                objects={
                    'depth': (depth, ),
                    'f_px': (f_px, )
                },
                tsk_type=EStoreObjectType.FOCAL_LENGTH | EStoreObjectType.DEPTH,
                op_type=EStoreOperatorType.INSERT
            )

        return TaskStatus.SUCCESS.value

    async def update_recon_object_by_task_id(
        self,
        task_id: str, 
        image: NDArray[np.uint8], 
        pose: NDArray[np.float32], 
        k_image: NDArray[np.float32]
    ):
        ...

    def _update_recon_object_by_task_id(
        self, 
        task_id: str, 
        image: NDArray[np.uint8], 
        pose: NDArray[np.float32], 
        k_image: NDArray[np.float32]
    ) -> int:
        if not self.container.exists(task_id):
            logging.warning(f"Task was aborted. Cannot find task_id: {task_id}, create new task_id before add pose data.")
            return TaskStatus.ABORTED.value
        
        update_objects = {
            "image":   image,
            "pose":    pose,
            "k_image": k_image,
            "k_depth": k_image, # TODO: k_depth is same as k_image for now
        }

        return self._update_objects_by_task_id(task_id, update_objects)
    
    def inference_depth_and_f_px(self, image: NDArray[np.uint8]) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        return self._inference_depth_and_f_px_impl(image)

    def _inference_depth_and_f_px_impl(self, image: NDArray[np.uint8]) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        depth, f_px = self.depth_predictor.infer(image)
        return (
            depth.to('cpu', dtype=torch.float32).numpy(), 
            f_px .to('cpu', dtype=torch.float32).numpy()
        )

    def inference_recon(self, task_id: str) -> Tuple[TaskStatus, Optional[Any]]:
        return self._inference_recon_impl(task_id)

    def _inference_recon_impl(self, task_id: str) -> Tuple[TaskStatus, Optional[Any]]:
        if not self.container.exists(task_id):
            logging.warning(f"Task was aborted. Cannot find task_id: {task_id}, create new task_id before infer recon.")
            return TaskStatus.ABORTED.value
        
        store = self.container.load_object_by_task_id(
            task_id, 
            EStoreObjectType.RECON_INFER_OBJECT
        )

        image_container   : ChunkArrayConcurrent | None = store["image"       ] or None
        depth_container   : ChunkArrayConcurrent | None = store["depth"       ] or None
        pose_container    : ChunkArrayConcurrent | None = store["pose"        ] or None
        k_image_container : ChunkArrayConcurrent | None = store["k_image"     ] or None
        k_depth_container : ChunkArrayConcurrent | None = store["k_depth"     ] or None

        assert (image_container   is not None, "Image container is None, but must be initialized already."  )
        assert (depth_container   is not None, "Depth container is None, but must be initialized already."  )
        assert (pose_container    is not None, "Pose container is None, but must be initialized already."   )
        assert (k_image_container is not None, "K_Image container is None, but must be initialized already.")
        assert (k_depth_container is not None, "K_Depth container is None, but must be initialized already.")

        if (len(image_container) == 0) or (len(depth_container) == 0) or (len(pose_container) == 0):
            logging.warning(f"Image/Depth/Pose data is empty. Cannot infer recon for task_id: {task_id}")
            return TaskStatus.ABORTED.value

        if not (len(image_container) == len(depth_container) == len(pose_container) == len(k_image_container) == len(k_depth_container)):
            logging.warning(f"Image/Depth/Pose/K_Image/K_Depth/Focal_Length data length mismatch. Cannot infer recon for task_id: {task_id}")
            return TaskStatus.ABORTED.value
        
        logging.info(f"Start 3D reconstruction for task_id: {task_id}, total frames: {len(image_container)}")
        logging.info(f" - image_container.len  : {len(image_container)}"  )
        logging.info(f" - depth_container.len  : {len(depth_container)}"  )
        logging.info(f" - pose_container.len   : {len(pose_container)}"   )
        logging.info(f" - k_image_container.len: {len(k_image_container)}")
        logging.info(f" - k_depth_container.len: {len(k_depth_container)}")

        recon_device = self.recon_predictor.config.device
        images   = torch.tensor(image_container  .get_raw_objects(), dtype=torch.uint8  , device=recon_device)
        depths   = torch.tensor(depth_container  .get_raw_objects(), dtype=torch.float32, device=recon_device)
        poses    = torch.tensor(pose_container   .get_raw_objects(), dtype=torch.float32, device=recon_device)
        k_images = torch.tensor(k_image_container.get_raw_objects(), dtype=torch.float32, device=recon_device)
        k_depths = torch.tensor(k_depth_container.get_raw_objects(), dtype=torch.float32, device=recon_device)
        
        # Create ReconPredictor batch data
        batch = {
            "images":   images,
            "depths":   depths,
            "poses":    poses,
            "k_images": k_images,
            "k_depths": k_depths
        }

        glb_bytes = self.recon_predictor.infer(
            batch=batch,
            task_id=task_id,
        )

        return (TaskStatus.SUCCESS.value, glb_bytes)

    # def _update_objects_by_task_id(
    #     self, 
    #     task_id: str,
    #     object: Dict[str, NDArray]
    # ) -> int:
    #     if not self.container.exists(task_id):
    #         logging.warning(f"Task was aborted. Cannot find task_id: {task_id}, create new task_id before infer depth.")
    #         return TaskStatus.ABORTED.value
        
    #     keys = object.keys()
    #     if (keys is None) or (len(keys) == 0):
    #         logging.warning(f"Object data is empty. Cannot update object for task_id: {task_id}")
    #         return TaskStatus.NOT_MODIFIED.value
        
    #     for key in keys:
    #         self.container.update_object_by_task_id(
    #             task_id=task_id, object=object[key],
    #             tsk_type=EStoreObjectType.to_enum(key), 
    #             op_type=EStoreOperatorType.INSERT
    #         )

    #     return TaskStatus.SUCCESS.value