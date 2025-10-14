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

from .utils.utils import NDArrayBuf, bench_mark
from .utils.verify import (
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

infer_router = APIRouter(prefix="/api/infer/recon")

def get_arvane_engine(request: Request) -> ArvaneEngine:
    return cast(ArvaneEngine, request.app.state.engine)

def get_recon_inference(request: Request) -> ReconPredictor:
    return cast(ReconPredictor, request.app.state.engine.recon_predictor)

@infer_router.options("")
async def preflight() -> Response:
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@infer_router.post("")
async def infer_recon(
    request: Request,
    arvane_engine: ArvaneEngine = Depends(get_arvane_engine),
    recon_predictor: ReconPredictor = Depends(get_recon_inference),
):
    ...