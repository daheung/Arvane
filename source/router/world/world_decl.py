from pydantic import BaseModel
from typing import Optional, List
from dataclasses import dataclass

class WorldCreatePayload(BaseModel):
    user_id: str
    name: Optional[str] = None

class WorldDeletePayload(BaseModel):
    name: Optional[str] = None
    task_id: str
    
    auto_update_depth: Optional[bool] = False

@dataclass
class WorldUpdatePayloadColor:
    buffer_b64: str
    shape: Optional[List[int]] = None
    dtype: Optional[str] = "uint8"
    order: Optional[str] = "C"
    endian: Optional[str] = "Little"
    urisafe : Optional[bool] = False
    data_url: Optional[bool] = False

class WorldUpdatePayload(BaseModel):
    name: Optional[str] = None
    task_id: str
    
    color: WorldUpdatePayloadColor
    k_color: List[float]
    pose: List[float]
    gt_origin: Optional[List[float]] = None
    gt_maxbound: Optional[List[float]] = None

    timestamp: int
    auto_update_depth: Optional[bool] = False

class WorldStartPayload(BaseModel):
    name: Optional[str] = None
    task_id: str

def verify_world_update(payload: WorldUpdatePayload) -> None:
    if payload.task_id is None or payload.task_id.strip() == "":
        raise ValueError("task_id must be provided and cannot be empty.")

    if payload.color.buffer_b64 is None:
        raise ValueError("color.buffer_b64 must not be null")
    
    if payload.k_color is not None and len(payload.k_color) != 9:
        raise ValueError("k_color must be a list of 9 floats.")

    if payload.pose is not None and len(payload.pose) != 16:
        raise ValueError("pose must be a list of 16 floats.")