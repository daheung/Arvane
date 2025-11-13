import cv2
import sys
import time
import base64
import logging
import numpy as np

from PIL import Image
from typing import cast
from fastapi import status
from fastapi import APIRouter, Request, Depends
from fastapi.responses import Response, JSONResponse
from fastapi.exceptions import HTTPException

from source.engine.arvane_engine import ArvaneEngine
from source.predictor.depth.predictor import DepthPredictor

from source.router.utils.utils import NDArrayBuf, bench_mark
from source.router.utils.verify import (
    check_dtype, 
    InvalidRequestQueryType, 
    InvalidRequestQueryException,
    InvalidRequestContentTypeException,
    InvalidRequestBodyException,
)

from source.engine.arvane_engine import ArvaneEngine
from source.runtime.infer.store import (
    InferenceChunkStoreConcurrent,
    EStoreOperatorType,
    EStoreObjectType,
)

from .debug_decl import DepthUpdatePayload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def get_arvane_engine(request: Request) -> ArvaneEngine:
    return cast(ArvaneEngine, request.app.state.engine)

def get_arvane_container(request: Request) -> InferenceChunkStoreConcurrent:
    return cast(InferenceChunkStoreConcurrent, request.app.state.engine.container)

debug_router = APIRouter(prefix="/api/debug")

@debug_router.options("")
async def preflight() -> Response:
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@debug_router.post("/depth")
async def depth(
    payload: DepthUpdatePayload,
    arvane_engine: ArvaneEngine = Depends(get_arvane_engine),
) -> Response:
    task_id = payload.task_id
    buffer_b64 = payload.depth.buffer_b64
    depth = base64.b64decode(buffer_b64)
    depth = np.frombuffer(depth, dtype=np.uint8)
    depth = cv2.imdecode(depth, cv2.IMREAD_UNCHANGED)

    arvane_engine.container.update_object_by_task_id(
        task_id,
        objects={
            'depth': (depth, )
        },
        tsk_type=EStoreObjectType.DEPTH
    )

    return JSONResponse({ "result": "ok" }, status_code=status.HTTP_200_OK)