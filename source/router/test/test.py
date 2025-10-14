import cv2
import sys
import time
import logging
import numpy as np

from typing import cast
from fastapi import status
from fastapi import APIRouter, Request, Depends
from fastapi.responses import Response, JSONResponse
from fastapi.exceptions import HTTPException

from source.router.utils.utils import NDArrayBuf, bench_mark
from source.router.utils.verify import (
    check_dtype, 
    InvalidRequestQueryType, 
    InvalidRequestQueryException,
    InvalidRequestContentTypeException,
    InvalidRequestBodyException,
)

from source.engine.arvane_engine import ArvaneEngine
from source.predictor.recon.predictor import ReconPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

container_router = APIRouter(prefix="/api/test/container")

def get_arvane_engine(request: Request):
    return cast(ArvaneEngine, request.app.state.engine)

@container_router.options("")
async def preflight() -> Response:
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@container_router.post("/push")
async def test_container(
    request: Request,
    recon_predictor: ArvaneEngine = Depends(get_arvane_engine)
):
    ...