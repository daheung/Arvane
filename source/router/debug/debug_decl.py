from pydantic import BaseModel
from dataclasses import dataclass

@dataclass
class DepthUpdatePayloadDepth:
    buffer_b64: str

class DepthUpdatePayload(BaseModel):
    task_id: str
    depth: DepthUpdatePayloadDepth