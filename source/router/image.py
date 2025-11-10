import cv2
import sys
import time
import logging
import numpy as np

from typing import cast, Optional
from fastapi import status
from fastapi import APIRouter, Request, Depends
from fastapi.responses import Response, JSONResponse
from datetime import datetime, time as dtime
from pydantic import BaseModel, Field

from source.engine.arvane_engine import ArvaneEngine
from source.predictor.recon.predictor import ReconPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

image_router = APIRouter(prefix="/api/image")

def get_arvane_engine(request: Request) -> ArvaneEngine:
    return cast(ArvaneEngine, request.app.state.engine)

def get_recon_inference(request: Request) -> ReconPredictor:
    return cast(ReconPredictor, request.app.state.engine.recon_predictor)

class ImagePayload(BaseModel):
    x: float
    y: float
    z: float
    # rotation: Annotated[List[float], Field(min_length=3, max_length=4)]
    height: Optional[float] = None
    time: datetime
    image: str

    # distance: Optional[float] = None
    pitch: float
    yaw: float
    roll: float

@image_router.options("")
async def preflight() -> Response:
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@image_router.post("")
async def default(payload: ImagePayload, request: Request):
    try:
        height = payload.height or 0.0
        logging.info(f"Received position ({payload.x:.2f}, {payload.y:.2f}, {payload.z:.2f}), height {height:.2f} at time {payload.time.isoformat()}")
        # Here you can add code to process the image data in payload.image

        return JSONResponse(content="ok", status_code=status.HTTP_200_OK)
    except Exception as e:
        logging.error(f"Error processing image data: {e}")
        return JSONResponse(content=e, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)