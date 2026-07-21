import cv2
import box
import torch
import logging
import numpy as np
import open3d as o3d

from tqdm import tqdm
from functools import partial
from numpy.typing import NDArray
from fastapi import BackgroundTasks
from open3d.geometry import TriangleMesh
from fastapi.concurrency import run_in_threadpool
from typing import Any, Tuple, Dict, List, Optional, TypeVar, Generic, Callable

from source.utils import load_config
from source.predictor.recon.predictor import ReconPredictor 
from source.predictor.depth.predictor import DepthPredictor
from source.predictor.ptv3.predictor import ExtractPredictor
from source.predictor.recon.utils import estimate_volume_bounds_from_recon_datas
from source.runtime.infer.store import InferenceChunkStoreConcurrent
from source.runtime.infer.defines import StoreStatus, EStoreObjectType, EStoreOperatorType
from source.engine.defines import TaskStatus
from source.runtime.container.array import ChunkArrayConcurrent
from source.runtime.executor.infer_task_executor import InferenceThreadExecutor
from source.runtime.device.device_manager import DeviceManager, DeviceDescriptor
from source.predictor.utils import (
    split_mesh_by_vertex_color,
    points_and_colors_to_mesh,
    glb_bytes_to_o3d_mesh,
    trimesh_dict_to_o3d
)

class ArvaneEngine:
    depth_predictor: DepthPredictor
    recon_predictor: ReconPredictor
    extract_predictor: ExtractPredictor

    container: InferenceChunkStoreConcurrent

    proxy_executor: InferenceThreadExecutor
    device_manager: DeviceManager

    def __init__(self):
        logging.info("Initializing ArvaneEngine ...")

        # init device manater
        self.device_manager = DeviceManager(enable_cpu=True)

        # init predictors
        depth_predictor, recon_predictor, extract_predictor = self.__predictor_init()

        # set app state dependencies
        self.depth_predictor = depth_predictor
        self.recon_predictor = recon_predictor
        self.extract_predictor = extract_predictor
        # load container
        self.container = InferenceChunkStoreConcurrent(chunk_size=64)

        # init proxy executor
        self.proxy_executor = InferenceThreadExecutor()

    def __predictor_init(self) -> Tuple[DepthPredictor, ReconPredictor, ExtractPredictor]:
        # load configuraion depth and resonstruction model.
        depth_config, recon_config = load_config()

        depth_device_descriptor: DeviceDescriptor = self.device_manager.get_device_considering_slack(required_minimum_memory_mib=6144)
        depth_config.device = depth_device_descriptor.device
        depth_predictor = DepthPredictor(depth_config)
        depth_predictor.init()

        recon_device_descriptor: DeviceDescriptor = self.device_manager.get_device_considering_slack(required_minimum_memory_mib=16384)
        recon_config.device = recon_device_descriptor.device
        recon_predictor = ReconPredictor(recon_config)
        recon_predictor.init()

        extract_device_descriptor: DeviceDescriptor = self.device_manager.get_device_considering_slack()
        extract_predictor = ExtractPredictor(model_name="sonata", device=extract_device_descriptor.device)
        extract_predictor.init()

        total_memory_per_mib = int(depth_device_descriptor.memory.total_memory / (1024 ** 2))
        free_memory_per_mib = int(depth_device_descriptor.memory.free_memory / (1024 ** 2))
        logging.info(f"Depth inference model binded device {depth_device_descriptor.description}.")
        logging.info(f"Binded device slot: {depth_device_descriptor.device_index}.")
        logging.info(f"  Device total memory: {total_memory_per_mib} Mib.")
        logging.info(f"  Device free  memory: {free_memory_per_mib} Mib")
        
        total_memory_per_mib = int(recon_device_descriptor.memory.total_memory / (1024 ** 2))
        free_memory_per_mib = int(recon_device_descriptor.memory.free_memory / (1024 ** 2))
        logging.info(f"Reconstruction inference model binded device {recon_device_descriptor.description}.")
        logging.info(f"Binded device slot: {recon_device_descriptor.device_index}.")
        logging.info(f"  Device total memory: {total_memory_per_mib} Mib.")
        logging.info(f"  Device free  memory: {free_memory_per_mib} Mib.")

        total_memory_per_mib = int(extract_device_descriptor.memory.total_memory / (1024 ** 2))
        free_memory_per_mib = int(extract_device_descriptor.memory.free_memory / (1024 ** 2))
        logging.info(f"Extraction inference model binded device {extract_device_descriptor.description}.")
        logging.info(f"Binded device slot: {extract_device_descriptor.device_index}.")
        logging.info(f"  Device total memory: {total_memory_per_mib} Mib.")
        logging.info(f"  Device free  memory: {free_memory_per_mib} Mib.")

        return depth_predictor, recon_predictor, extract_predictor
    
    async def run_process(self, task_id: str):
        try:
            self.container[task_id].store_status = StoreStatus.DEPTH
            depth_status: TaskStatus = await self.proxy_executor.execute(
                self._update_depth_object_by_task_id, 
                task_id
            )
            if (
                depth_status != TaskStatus.SUCCESS and
                depth_status != TaskStatus.NOT_MODIFIED
            ):
                logging.warning(f"Auto-update depth failed (bg) for task_id={task_id}, status={depth_status}")
                self.container[task_id].depth_container = ChunkArrayConcurrent()
                self.container[task_id].recon_object = None
                return
            
            user_id = self.container[task_id].user_id
            self.container[task_id].store_status = StoreStatus.RECON
            result: Tuple[TaskStatus, Any] = await self.proxy_executor.execute(
                self._inference_recon_impl,
                user_id, 
                task_id
            )
            recon_status, glb_bytes = result

            if (recon_status != TaskStatus.SUCCESS):
                logging.warning(f"Auto-update reconstruction failed (bg) for task_id={task_id}, status={recon_status}")
                self.container[task_id].store_status = StoreStatus.ABORTED
                self.container[task_id].recon_object = None
                return

            convert_o3d_mesh_func = glb_bytes_to_o3d_mesh if isinstance(glb_bytes, bytes) else trimesh_dict_to_o3d
            recon_mesh: TriangleMesh = convert_o3d_mesh_func(glb_bytes)

            self.container[task_id].store_status = StoreStatus.DONE
            self.container[task_id].recon_object = recon_mesh
            
            extraction_result: Tuple[TaskStatus, torch.Tensor, Optional[torch.Tensor]] = await self.inference_extraction(
                recon_mesh
            )
            extraction_status, points, color = extraction_result
            
            if (extraction_status != TaskStatus.SUCCESS):
                logging.warning(f"Auto-update reconstruction failed (bg) for task_id={task_id}, status={recon_status}")
                self.container[task_id].store_status = StoreStatus.ABORTED
                self.container[task_id].extraction_object = None
                return
            
            recon_object_considering_extraction: TriangleMesh = points_and_colors_to_mesh(points, color)
            self.container[task_id].recon_object = recon_object_considering_extraction

            extraction_mesh: List[TriangleMesh] = split_mesh_by_vertex_color(
                recon_object_considering_extraction,
                color_eps=0.0001,
            )
            self.container[task_id].extraction_object = extraction_mesh
            logging.info(f"Auto-update reconstruction and extraction done (bg) for task_id={task_id}")

        except Exception as e:
            self.container[task_id].store_status = StoreStatus.ABORTED
            logging.exception(f"Auto-update crashed (bg) for task_id={task_id}: {e}")

        finally:
            ...
    
    async def update_depth_object_by_task_id(self, task_id: str):
        try:
            status = await self.proxy_executor.execute(self._update_depth_object_by_task_id, task_id)
            if status != TaskStatus.SUCCESS:
                logging.warning(f"Auto-update depth failed (bg) for task_id={task_id}, status={status.name}")
            else:
                logging.info(f"Auto-update depth done (bg) for task_id={task_id}")

        except Exception as e:
            logging.exception(f"Auto-update depth crashed (bg) for task_id={task_id}: {e}")
    
    def _update_depth_object_by_task_id(self, task_id: str) -> TaskStatus:
        if not self.container.exists(task_id):
            logging.warning(f"Task was aborted. Cannot find task_id: {task_id}, create new task_id before infer depth.")
            return TaskStatus.ABORTED
        
        store = self.container[task_id]

        image_container  : Optional[ChunkArrayConcurrent] = store.image_container
        k_image_container: Optional[ChunkArrayConcurrent] = store.k_image_container
        depth_container  : Optional[ChunkArrayConcurrent] = store.depth_container

        assert depth_container is not None, "Depth container is None, but must be initialized already."

        if (image_container is None):
            logging.warning(f"Image data is empty. Cannot infer depth for task_id: {task_id}")
            return TaskStatus.ABORTED

        num_depth: int = len(depth_container)
        num_image: int = len(image_container)
        # case 1) depth_container == image_container:
        #  더 이상 할 작업이 없음.
        # case 2) depth_container >  image_container:
        #  작업상 depth_container가 image_container보다 더 클 수 없음;
        #  같이 이미지를 중복 복원 or 배열의 일부 부분이 비어있을 가능성 있음
        if (num_depth >= num_image):
            logging.info(f"Depth data already exists. Skip depth inference for task_id: {task_id}")
            return TaskStatus.ABORTED
        

        offset_start = num_image - num_image
        for idx in tqdm(range(offset_start, num_image)):
            k_color: NDArray = k_image_container[idx].object
            fx, fy = k_color[0, 0], k_color[1, 1]
            depth, f_px = self._inference_depth_and_f_px_impl(
                image_container[idx].object,
                (fx + fy) / 2
            )

            timestamp = store.image_container[idx].key
            self.container.update_object_by_task_id(
                task_id=task_id,
                objects={
                    'depth': (depth, ),
                    'f_px': (f_px, )
                },
                tsk_type=EStoreObjectType.FOCAL_LENGTH | EStoreObjectType.DEPTH,
                timestamp=timestamp
            )

        return TaskStatus.SUCCESS
        
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
            return TaskStatus.ABORTED
        
        update_objects = {
            "image":   image,
            "pose":    pose,
            "k_image": k_image,
            "k_depth": k_image, # TODO: k_depth is same as k_image for now
        }

        return self._update_objects_by_task_id(task_id, update_objects)
    
    def inference_depth_and_f_px(
        self, 
        image: NDArray[np.uint8],
        f_px: NDArray[np.float32]
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        return self._inference_depth_and_f_px_impl(image, f_px)

    def _inference_depth_and_f_px_impl(
        self, 
        image: NDArray[np.uint8],
        f_px: NDArray[np.float32]
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        # Pytorch (C, H, W) -> OpenCV (H, W, C)
        # 이미지를 모델에 넣을 수 있는 크기로 전처리 [width: 640, height: 480]
        image = cv2.resize(
            image.transpose((1, 2, 0)), 
            (640, 480), 
            interpolation=cv2.INTER_AREA
        )

        depth, f_px = self.depth_predictor.infer(image, f_px)
        return (
            depth.to('cpu', dtype=torch.float32).numpy(), 
            f_px .to('cpu', dtype=torch.float32).numpy()
        )

    def inference_recon(self, user_id: str, task_id: str) -> Tuple[TaskStatus, Optional[Any]]:
        return self._inference_recon_impl(user_id, task_id)

    def _inference_recon_impl(
        self, 
        user_id: str, 
        task_id: str
    ) -> Tuple[TaskStatus, Optional[Any]]:
        if not self.container.exists(task_id):
            logging.warning(f"Task was aborted. Cannot find task_id: {task_id}, create new task_id before infer recon.")
            return TaskStatus.ABORTED
        
        store = self.container[task_id]
        image_container   : ChunkArrayConcurrent | None = store.image_container   or None
        depth_container   : ChunkArrayConcurrent | None = store.depth_container   or None
        pose_container    : ChunkArrayConcurrent | None = store.pose_container    or None
        k_image_container : ChunkArrayConcurrent | None = store.k_image_container or None
        k_depth_container : ChunkArrayConcurrent | None = store.k_depth_container or None

        assert image_container   is not None, "Image container is None, but must be initialized already."  
        assert depth_container   is not None, "Depth container is None, but must be initialized already."  
        assert pose_container    is not None, "Pose container is None, but must be initialized already."   
        assert k_image_container is not None, "K_Image container is None, but must be initialized already."
        assert k_depth_container is not None, "K_Depth container is None, but must be initialized already."

        if (len(image_container) == 0) or (len(depth_container) == 0) or (len(pose_container) == 0):
            logging.warning(f"Image/Depth/Pose data is empty. Cannot infer recon for task_id: {task_id}")
            return TaskStatus.ABORTED

        if not (len(image_container) == len(depth_container) == len(pose_container) == len(k_image_container) == len(k_depth_container)):
            logging.warning(f"Image/Depth/Pose/K_Image/K_Depth/Focal_Length data length mismatch. Cannot infer recon for task_id: {task_id}")
            return TaskStatus.ABORTED
        
        logging.info(f"Start 3D reconstruction for task_id: {task_id}, total frames: {len(image_container)}")
        logging.info(f"  Image_container.len  : {len(image_container)}"  )
        logging.info(f"  Depth_container.len  : {len(depth_container)}"  )
        logging.info(f"  Pose_container.len   : {len(pose_container)}"   )
        logging.info(f"  K_image_container.len: {len(k_image_container)}")
        logging.info(f"  K_depth_container.len: {len(k_depth_container)}")
        
        recon_device: torch.device = torch.device(self.recon_predictor.config.device)
        images  : NDArray = np.array(image_container  .get_raw_objects(), dtype=np.uint8  )
        depths  : NDArray = np.array(depth_container  .get_raw_objects(), dtype=np.float32)
        poses   : NDArray = np.array(pose_container   .get_raw_objects(), dtype=np.float64)
        k_images: NDArray = np.array(k_image_container.get_raw_objects(), dtype=np.float32)
        k_depths: NDArray = np.array(k_depth_container.get_raw_objects(), dtype=np.float32)

        TARGET_WIDTH, TARGET_HEIGHT = (640, 480)
        _, _, imheight, imwidth = images.shape
        k_images = k_images[0]
        k_images[0] *= TARGET_WIDTH / imwidth
        k_images[1] *= TARGET_HEIGHT / imheight
        k_images: NDArray = np.array([k_images for _ in range(len(k_image_container))])

        # _, dpheight, dpwidth = depths.shape
        k_depths = k_depths[0]
        k_depths[0] *= TARGET_WIDTH / imwidth
        k_depths[1] *= TARGET_HEIGHT / imheight
        k_depths: NDArray = np.array([k_depths for _ in range(len(k_depth_container))])

        images = images.transpose((0, 2, 3, 1))
        images = [cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_NEAREST) for image in images]
        images = np.stack(images, axis=0).transpose((0, 3, 1, 2)) / 255
        images = images.astype(dtype=np.float32)[:, None, ...]

        depths = [cv2.resize(depth, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_NEAREST) for depth in depths]
        depths = np.stack(depths, axis=0)
        depths = depths.astype(dtype=np.float32)[:, None, ...]

        with torch.no_grad():
            _, gt_origin, gt_maxbound = estimate_volume_bounds_from_recon_datas(
                depths, 
                poses, 
                k_images[0],
                device=recon_device,
                ret_tsdf=False
            )
        
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # Create ReconPredictor batch data
        batch = {
            "images"     : torch.tensor(images  , dtype=torch.float32),
            "depths"     : torch.tensor(depths  , dtype=torch.float32),
            "poses"      : torch.tensor(poses   , dtype=torch.float32),
            "k_image"    : torch.tensor(k_images, dtype=torch.float32),
            "k_depth"    : torch.tensor(k_depths, dtype=torch.float32),
            "gt_origin"  : gt_origin  [None, ...],
            "gt_maxbound": gt_maxbound[None, ...]
        }

        log = store.recon_log
        with torch.no_grad():
            glb_bytes = self.recon_predictor.infer(
                batch=batch,
                user_id=user_id,
                task_id=task_id,
                device=recon_device,
                log=log,
            )

        return (TaskStatus.SUCCESS, glb_bytes)

    async def inference_extraction(
        self, 
        glb_bytes: TriangleMesh
    ) -> Tuple[TaskStatus, torch.Tensor, Optional[torch.Tensor]]:
        return self._inference_extraction_impl(glb_bytes)
    
    def _inference_extraction_impl(
        self, 
        glb_bytes: TriangleMesh
    ) -> Tuple[TaskStatus, torch.Tensor, Optional[torch.Tensor]]:
        with torch.no_grad():
            result: Tuple[torch.Tensor, Optional[torch.Tensor]] = self.extract_predictor.infer(
                glb_bytes_or_file=glb_bytes,
                return_color=True
            )

        return (TaskStatus.SUCCESS, *result)

    # def _update_objects_by_task_id(
    #     self, 
    #     task_id: str,
    #     object: Dict[str, NDArray]
    # ) -> int:
    #     if not self.container.exists(task_id):
    #         logging.warning(f"Task was aborted. Cannot find task_id: {task_id}, create new task_id before infer depth.")
    #         return TaskStatus.ABORTED
        
    #     keys = object.keys()
    #     if (keys is None) or (len(keys) == 0):
    #         logging.warning(f"Object data is empty. Cannot update object for task_id: {task_id}")
    #         return TaskStatus.NOT_MODIFIED
        
    #     for key in keys:
    #         self.container.update_object_by_task_id(
    #             task_id=task_id, object=object[key],
    #             tsk_type=EStoreObjectType.to_enum(key), 
    #             op_type=EStoreOperatorType.INSERT
    #         )

    #     return TaskStatus.SUCCESS