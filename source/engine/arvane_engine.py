import os
import sys
import time
import torch
import logging
import threading
import numpy as np

from typing import Any, Tuple, Dict, List, Optional, TypeVar, Generic, Callable
from numpy.typing import NDArray

from source.utils import load_config
from source.predictor.recon.predictor import ReconPredictor 
from source.predictor.depth.predictor import DepthPredictor
from source.runtime.infer.store import InferenceStore, InferenceChunkStoreConcurrent
from source.runtime.infer.defines import EStoreObjectType, EStoreOperatorType
from source.engine.defines import TaskStatus
from source.runtime.array import ChunkArrayConcurrent

class ArvaneEngine:
    depth_predictor: DepthPredictor
    recon_predictor: ReconPredictor

    container: InferenceChunkStoreConcurrent

    def __init__(self):
        logging.info("ArvaneEngine initialized.")
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
    
    def update_depth_object_by_task_id(self, task_id: str) -> TaskStatus:
        if not self.container.exists(task_id):
            logging.warning(f"Task was aborted. Cannot find task_id: {task_id}, create new task_id before infer depth.")
            return TaskStatus.ABORTED.value
        
        store = self.container.load_object_by_task_id(task_id, EStoreObjectType.IMAGE, EStoreObjectType.DEPTH)
        depth_container: ChunkArrayConcurrent | None = store["depth"] or None
        image_container: ChunkArrayConcurrent | None = store["image"] or None

        assert (depth_container is not None, "Depth container is None, but must be initialized already.")

        if (image_container is None):
            logging.warning(f"Image data is empty. Cannot infer depth for task_id: {task_id}")
            return TaskStatus.ABORTED.value

        if (len(depth_container) == len(image_container)):
            logging.info(f"Depth data already exists. Skip depth inference for task_id: {task_id}")
            return TaskStatus.NOT_MODIFIED.value
        
        depth_data: List[NDArray[np.float32]] = []
        f_px_data : List[NDArray[np.float32]] = []

        for idx in range(len(depth_container)):
            depth, f_px = self._inference_depth_and_f_px_impl(image_container[idx])

            depth_data.append(depth)
            f_px_data.append(f_px)

        self.container.object_update(
            task_id=task_id,
            object=depth_data,
            tsk_type=EStoreObjectType.DEPTH,
            op_type=EStoreOperatorType.INSERT
        )

        self.container.object_update(
            task_id=task_id,
            object=f_px_data,
            tsk_type=EStoreObjectType.FOCAL_LENGTH,
            op_type=EStoreOperatorType.INSERT
        )

        return TaskStatus.SUCCESS.value
    
    def update_recon_object_by_task_id(
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
            depth.to('cpu', dtype=np.float32).numpy(), 
            f_px .to('cpu', dtype=np.float32).numpy()
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

    def _update_objects_by_task_id(
        self, 
        task_id: str,
        object: Dict[str, NDArray]
    ) -> int:
        if not self.container.exists(task_id):
            logging.warning(f"Task was aborted. Cannot find task_id: {task_id}, create new task_id before infer depth.")
            return TaskStatus.ABORTED.value
        
        keys = object.keys()
        if (keys is None) or (len(keys) == 0):
            logging.warning(f"Object data is empty. Cannot update object for task_id: {task_id}")
            return TaskStatus.NOT_MODIFIED.value
        
        for key in keys:
            self.container.update_object_by_task_id(
                task_id=task_id, object=object[key],
                tsk_type=EStoreObjectType.to_enum(key), 
                op_type=EStoreOperatorType.INSERT
            )

        return TaskStatus.SUCCESS.value