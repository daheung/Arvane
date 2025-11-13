import torch
import numpy as np

from enum import IntFlag
from typing import Any
from numpy.typing import NDArray
from dataclasses import dataclass
from source.runtime.container.array import ChunkArrayConcurrent

@dataclass
class ReconLog:
    init_time_0       : int = 0
    init_time_1       : int = 0
    per_view_time     : int = 0
    n_views           : int = 0
    n_inits           : int = 0
    final_step_time_0 : int = 0
    final_step_time_1 : int = 0
    n_final_steps     : int = 0

@dataclass
class StoreStatus:
    CREATED = 0x00
    DEPTH = 0x01
    RECON = 0x02
    EXTRACT = 0x04
    
    DONE = 0x0A
    ABORTED = 0x0FD
    PENDING_KILL = 0xFF

@dataclass
class ReconStore:
    M                       : torch.Tensor = None
    running_count           : torch.Tensor = None
    running_density         : torch.Tensor = None
    running_tsdf            : torch.Tensor = None
    global_step             : torch.Tensor = None
    global_coords           : torch.Tensor = None
    running_density_weight  : torch.Tensor = None
    running_tsdf_weight     : torch.Tensor = None
        
class InferenceStore:
    user_id: str
    task_id: str
    chunk_size: int
    store_status: StoreStatus

    depth_container       : ChunkArrayConcurrent[NDArray[np.float32]]
    pose_container        : ChunkArrayConcurrent[NDArray[np.float32]]
    image_container       : ChunkArrayConcurrent[NDArray[np.uint8  ]]
    k_image_container     : ChunkArrayConcurrent[NDArray[np.float32]]
    k_depth_container     : ChunkArrayConcurrent[NDArray[np.float32]]
    focal_length_container: ChunkArrayConcurrent[NDArray[np.float32]]

    recon_container: ReconStore
    recon_logging: ReconLog

    # 최종 복원된 3D Map Object
    recon_object: Any

    def __init__(self, user_id: str, task_id: str, chunk_size: int):
        self.user_id = user_id
        self.task_id = task_id
        self.chunk_size = chunk_size
        self.store_status = StoreStatus.CREATED

        # initialize containers
        self.depth_container        = ChunkArrayConcurrent(chunk_size=chunk_size)
        self.pose_container         = ChunkArrayConcurrent(chunk_size=chunk_size)
        self.image_container        = ChunkArrayConcurrent(chunk_size=chunk_size)
        self.k_image_container      = ChunkArrayConcurrent(chunk_size=chunk_size)
        self.k_depth_container      = ChunkArrayConcurrent(chunk_size=chunk_size)
        self.focal_length_container = ChunkArrayConcurrent(chunk_size=chunk_size)

        self.recon_container = ReconStore()
        self.recon_logging = ReconLog()
        
        self.recon_object = None

class EStoreOperatorType(IntFlag):
    INSERT = 0x01

class EStoreObjectType(IntFlag):
    IMAGE         = 0x001
    DEPTH         = 0x002
    POSE          = 0x004
    K_IMAGE       = 0x008
    K_DEPTH       = 0x010
    FOCAL_LENGTH  = 0x020
    RECON_STORE   = 0x040
    RECON_LOG     = 0x080
    RECON_OBJECT  = 0x100

    RECON_INFER_OBJECT = IMAGE | DEPTH | POSE | K_IMAGE | K_DEPTH
    RECON_INFER_NO_DEPTH = RECON_INFER_OBJECT & (~DEPTH)
    RECON_INFER_OBJECT_WITH_LOG = RECON_INFER_OBJECT | RECON_LOG
    ALL_OBJECTS = IMAGE | DEPTH | POSE | K_IMAGE | K_DEPTH | FOCAL_LENGTH | RECON_STORE | RECON_LOG | RECON_OBJECT

    def to_enum(value: str) -> 'EStoreObjectType':
        mapping = {
            "image"        : EStoreObjectType.IMAGE,
            "depth"        : EStoreObjectType.DEPTH,
            "pose"         : EStoreObjectType.POSE,
            "k_image"      : EStoreObjectType.K_IMAGE,
            "k_depth"      : EStoreObjectType.K_DEPTH,
            "focal_length" : EStoreObjectType.FOCAL_LENGTH,
            "recon_store"  : EStoreObjectType.RECON_STORE,
            "recon_log"    : EStoreObjectType.RECON_LOG,
            "recon_object" : EStoreObjectType.RECON_OBJECT,
        }

        return mapping.get(value.lower(), None)